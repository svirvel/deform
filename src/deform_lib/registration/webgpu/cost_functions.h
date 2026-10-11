#pragma once

#include <deform_lib/config.h>

#ifdef DF_USE_WEBGPU

#include "block_grid.h"
#include "displacement_field.h"
#include "../settings.h"

#include <stk/image/webgpu_volume.h>
#include <stk/math/float3.h>

#include <memory>
#include <vector>

// WebGPU versions of the cost functions in registration/gpu/cost_functions.
//
// Unlike the CUDA versions, which are invoked for a region of the volume, the
// costs are computed for all blocks that are assigned a slot in the provided
// cost buffers.

struct WebGpuCostFunction
{
    WebGpuCostFunction() {}
    virtual ~WebGpuCostFunction() {}

    // Costs are accumulated into the unary cost buffer (of type float2),
    //  where x is the cost before applying delta and y is the cost after.
    // df           : Displacement field
    // delta        : Delta applied to the displacement, typically based on the step-size.
    // cost         : Destination for cost
    virtual void cost(
        const WebGpuDisplacementField& df,
        const float3& delta,
        float weight,
        Settings::UpdateRule update_rule,
        webgpu::CostBuffers& cost
    ) = 0;

    void set_fixed_mask(const stk::WebGpuVolume& fixed_mask)
    {
        _fixed_mask = fixed_mask;
    }
    void set_moving_mask(const stk::WebGpuVolume& moving_mask)
    {
        _moving_mask = moving_mask;
    }

    stk::WebGpuVolume _fixed_mask;
    stk::WebGpuVolume _moving_mask;
};

struct WebGpuCostFunction_SSD : public WebGpuCostFunction
{
    WebGpuCostFunction_SSD(
        const stk::WebGpuVolume& fixed,
        const stk::WebGpuVolume& moving
    ) : _fixed(fixed), _moving(moving) {}

    void cost(
        const WebGpuDisplacementField& df,
        const float3& delta,
        float weight,
        Settings::UpdateRule update_rule,
        webgpu::CostBuffers& cost
    );

    stk::WebGpuVolume _fixed;
    stk::WebGpuVolume _moving;
};

struct WebGpuCostFunction_NCC : public WebGpuCostFunction
{
    WebGpuCostFunction_NCC(
        const stk::WebGpuVolume& fixed,
        const stk::WebGpuVolume& moving,
        int radius
    ) : _fixed(fixed), _moving(moving), _radius(radius) {}

    void cost(
        const WebGpuDisplacementField& df,
        const float3& delta,
        float weight,
        Settings::UpdateRule update_rule,
        webgpu::CostBuffers& cost
    );

    stk::WebGpuVolume _fixed;
    stk::WebGpuVolume _moving;
    int _radius;
};

struct WebGpuCostFunction_Landmarks : public WebGpuCostFunction
{
    WebGpuCostFunction_Landmarks(
        const std::vector<float3>& fixed_landmarks,
        const std::vector<float3>& moving_landmarks,
        const float decay
    );

    void cost(
        const WebGpuDisplacementField& df,
        const float3& delta,
        float weight,
        Settings::UpdateRule update_rule,
        webgpu::CostBuffers& cost
    );

    // Fixed landmarks followed by their displacements (float4)
    stk::webgpu::Buffer _landmarks;
    int _landmark_count;
    float _half_decay;
};

struct WebGpuCostFunction_SoftConstraints : public WebGpuCostFunction
{
    // constraint_mask   : float, non-zero for constrained voxels
    // constraint_values : float4
    WebGpuCostFunction_SoftConstraints(
        const stk::WebGpuVolume& constraint_mask,
        const stk::WebGpuVolume& constraint_values
    ) : _constraint_mask(constraint_mask), _constraint_values(constraint_values) {}

    void cost(
        const WebGpuDisplacementField& df,
        const float3& delta,
        float weight,
        Settings::UpdateRule update_rule,
        webgpu::CostBuffers& cost
    );

    stk::WebGpuVolume _constraint_mask;
    stk::WebGpuVolume _constraint_values;
};

// A set of weighted cost functions
class WebGpuUnaryFunction
{
public:
    struct WeightedFunction {
        float weight;
        std::unique_ptr<WebGpuCostFunction> function;
    };

    WebGpuUnaryFunction() {}
    ~WebGpuUnaryFunction() {}

    void operator()(
        const WebGpuDisplacementField& df,
        const float3& delta,
        Settings::UpdateRule update_rule,
        webgpu::CostBuffers& cost
    )
    {
        for (auto& fn : _functions) {
            fn.function->cost(df, delta, fn.weight, update_rule, cost);
        }
    }

    void add_function(std::unique_ptr<WebGpuCostFunction>& fn, float weight)
    {
        _functions.push_back({weight, std::move(fn)});
    }

private:
    std::vector<WeightedFunction> _functions;
};

class WebGpuBinaryFunction
{
public:
    WebGpuBinaryFunction() :
        _weight(0.0f),
        _scale(1.0f),
        _half_exponent(1.0f),
        _spacing{0, 0, 0}
    {
    }
    ~WebGpuBinaryFunction() {}

    void set_regularization_weight(const float weight)
    {
        _weight = weight;
    }
    void set_regularization_scale(const float scale)
    {
        _scale = scale;
    }
    void set_regularization_exponent(const float exponent)
    {
        _half_exponent = 0.5f * exponent;
    }
    void set_fixed_spacing(const float3& spacing)
    {
        _spacing = spacing;
    }

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_weight_map(const stk::WebGpuVolume& map) { _weight_map = map; }
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

    // Computes the regularization cost in three directions (x+, y+, z+), with and without
    //  applied delta. Results are stored into the binary cost buffers (of type float4),
    //  as {E00, E01, E10, E11}.
    // df           : Displacement field
    // delta        : Delta applied to the displacement, typically based on the step-size.
    void operator()(
        const WebGpuDisplacementField& df,
        const float3& delta,
        Settings::UpdateRule update_rule,
        webgpu::CostBuffers& cost
    );

private:
    float _weight;
    float _scale;
    float _half_exponent;
    float3 _spacing;

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    stk::WebGpuVolume _weight_map;
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP
};

#endif // DF_USE_WEBGPU
