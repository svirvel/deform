#include "cost_functions.h"

#ifdef DF_USE_WEBGPU

#include <stk/common/assert.h>
#include <stk/math/float4.h>

#include <cmath>

namespace {
    using stk::webgpu::Access_Read;
    using stk::webgpu::Access_ReadWrite;
    using stk::webgpu::Buffer;

    const Buffer& buffer_or_dummy(const stk::WebGpuVolume& vol)
    {
        return vol.valid() ? vol.buffer() : Buffer::dummy();
    }

    int update_rule_index(Settings::UpdateRule update_rule)
    {
        // Has to match the constants in the shaders
        return update_rule == Settings::UpdateRule_Compositive ? 1 : 0;
    }

    // Invokes a unary cost function
    // a, b     : Inputs specific to the cost function
    // moving   : Moving volume, or an invalid volume if not used
    void run_unary(
        webgpu::CostFunction fn,
        const WebGpuCostFunction& base,
        const WebGpuDisplacementField& df,
        const float3& delta,
        float weight,
        Settings::UpdateRule update_rule,
        webgpu::CostBuffers& cost,
        const Buffer& a,
        const Buffer& b,
        const stk::WebGpuVolume& moving,
        int iparam = 0,
        float fparam = 0.0f
    )
    {
        static std::unique_ptr<stk::webgpu::Kernel> kernels[4];
        if (!kernels[fn]) {
            kernels[fn].reset(new stk::webgpu::Kernel(
                webgpu::unary_shader(fn),
                {Access_Read, Access_Read, Access_Read, Access_Read, Access_Read,
                 Access_Read, Access_ReadWrite},
                sizeof(webgpu::UnaryParams)
            ));
        }

        FATAL_IF(base._fixed_mask.valid() && base._fixed_mask.voxel_type() != stk::Type_Float ||
                 base._moving_mask.valid() && base._moving_mask.voxel_type() != stk::Type_Float)
            << "Unsupported pixel type";

        webgpu::UnaryParams params = {};
        params.grid = cost.grid.params();
        params.df = df.params();

        if (moving.valid()) {
            const dim3 dims = moving.size();
            const float3 origin = moving.origin();
            const float3 spacing = moving.spacing();
            params.moving = webgpu::MovingParams{
                {int(dims.x), int(dims.y), int(dims.z), 0},
                {origin.x, origin.y, origin.z, 0},
                {1.0f / spacing.x, 1.0f / spacing.y, 1.0f / spacing.z, 0},
                {}
            };
            webgpu::set_rows(params.moving.inv_direction, moving.inverse_direction());
        }
        else {
            // Not used, but keep the transform well-defined
            params.moving.inv_spacing[0] = 1.0f;
            params.moving.inv_spacing[1] = 1.0f;
            params.moving.inv_spacing[2] = 1.0f;
            webgpu::set_rows(params.moving.inv_direction, Matrix3x3f::Identity);
        }

        params.delta[0] = delta.x;
        params.delta[1] = delta.y;
        params.delta[2] = delta.z;
        params.weight = weight;
        params.update_rule = update_rule_index(update_rule);
        params.has_fixed_mask = base._fixed_mask.valid() ? 1 : 0;
        params.has_moving_mask = base._moving_mask.valid() ? 1 : 0;
        params.iparams[0] = iparam;
        params.fparams[0] = fparam;

        kernels[fn]->run(
            &params,
            {
                cost.slots,
                df.volume().buffer(),
                a,
                b,
                buffer_or_dummy(base._fixed_mask),
                buffer_or_dummy(base._moving_mask),
                cost.unary
            },
            df.size()
        );
    }
}

void WebGpuCostFunction_SSD::cost(
    const WebGpuDisplacementField& df,
    const float3& delta,
    float weight,
    Settings::UpdateRule update_rule,
    webgpu::CostBuffers& cost
)
{
    FATAL_IF(_fixed.voxel_type() != stk::Type_Float ||
             _moving.voxel_type() != stk::Type_Float)
        << "Unsupported pixel type";

    run_unary(webgpu::CostFunction_SSD, *this, df, delta, weight, update_rule, cost,
              _fixed.buffer(), _moving.buffer(), _moving);
}

void WebGpuCostFunction_NCC::cost(
    const WebGpuDisplacementField& df,
    const float3& delta,
    float weight,
    Settings::UpdateRule update_rule,
    webgpu::CostBuffers& cost
)
{
    FATAL_IF(_fixed.voxel_type() != stk::Type_Float ||
             _moving.voxel_type() != stk::Type_Float)
        << "Unsupported pixel type";

    run_unary(webgpu::CostFunction_NCC, *this, df, delta, weight, update_rule, cost,
              _fixed.buffer(), _moving.buffer(), _moving, _radius);
}

WebGpuCostFunction_Landmarks::WebGpuCostFunction_Landmarks(
    const std::vector<float3>& fixed_landmarks,
    const std::vector<float3>& moving_landmarks,
    const float decay
) :
    _landmark_count(int(fixed_landmarks.size())),
    _half_decay(decay / 2.0f)
{
    ASSERT(fixed_landmarks.size() == moving_landmarks.size());
    ASSERT(!fixed_landmarks.empty());

    std::vector<float4> data;
    for (const float3& lm : fixed_landmarks) {
        data.push_back(float4{lm.x, lm.y, lm.z, 0});
    }
    for (size_t i = 0; i < fixed_landmarks.size(); ++i) {
        const float3 d = moving_landmarks[i] - fixed_landmarks[i];
        data.push_back(float4{d.x, d.y, d.z, 0});
    }

    _landmarks = Buffer(data.size() * sizeof(float4));
    _landmarks.upload(data.data(), data.size() * sizeof(float4));
}

void WebGpuCostFunction_Landmarks::cost(
    const WebGpuDisplacementField& df,
    const float3& delta,
    float weight,
    Settings::UpdateRule update_rule,
    webgpu::CostBuffers& cost
)
{
    run_unary(webgpu::CostFunction_Landmarks, *this, df, delta, weight, update_rule, cost,
              _landmarks, Buffer::dummy(), stk::WebGpuVolume(), _landmark_count, _half_decay);
}

void WebGpuCostFunction_SoftConstraints::cost(
    const WebGpuDisplacementField& df,
    const float3& delta,
    float weight,
    Settings::UpdateRule update_rule,
    webgpu::CostBuffers& cost
)
{
    FATAL_IF(_constraint_mask.voxel_type() != stk::Type_Float ||
             _constraint_values.voxel_type() != stk::Type_Float4)
        << "Unsupported pixel type";

    run_unary(webgpu::CostFunction_SoftConstraints, *this, df, delta, weight, update_rule,
              cost, _constraint_mask.buffer(), _constraint_values.buffer(),
              stk::WebGpuVolume());
}

void WebGpuBinaryFunction::operator()(
    const WebGpuDisplacementField& df,
    const float3& delta,
    Settings::UpdateRule update_rule,
    webgpu::CostBuffers& cost
)
{
    static stk::webgpu::Kernel kernel(
        webgpu::binary_shader(),
        {Access_Read, Access_Read, Access_Read, Access_ReadWrite, Access_ReadWrite,
         Access_ReadWrite},
        sizeof(webgpu::BinaryParams)
    );

    webgpu::BinaryParams params = {};
    params.grid = cost.grid.params();
    params.df = df.params();
    params.delta[0] = delta.x;
    params.delta[1] = delta.y;
    params.delta[2] = delta.z;
    params.inv_spacing2_exp[0] = 1.0f / std::pow(_spacing.x*_spacing.x, _half_exponent);
    params.inv_spacing2_exp[1] = 1.0f / std::pow(_spacing.y*_spacing.y, _half_exponent);
    params.inv_spacing2_exp[2] = 1.0f / std::pow(_spacing.z*_spacing.z, _half_exponent);
    params.weight = _weight;
    params.scale = _scale;
    params.half_exponent = _half_exponent;
    params.update_rule = update_rule_index(update_rule);

    stk::WebGpuVolume weight_map;
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    weight_map = _weight_map;
#endif
    params.has_weight_map = weight_map.valid() ? 1 : 0;

    kernel.run(
        &params,
        {
            cost.slots,
            df.volume().buffer(),
            buffer_or_dummy(weight_map),
            cost.binary[0],
            cost.binary[1],
            cost.binary[2]
        },
        df.size()
    );
}

#endif // DF_USE_WEBGPU
