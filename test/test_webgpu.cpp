#include "catch.hpp"

#include <deform_lib/config.h>

#ifdef DF_USE_WEBGPU

#include <deform_lib/filters/gaussian_filter.h>
#include <deform_lib/filters/resample.h>
#include <deform_lib/filters/webgpu/filters.h>

#include <deform_lib/make_unique.h>
#include <deform_lib/registration/affine_transform.h>
#include <deform_lib/registration/cost_functions/cross_correlation_sphere.h>
#include <deform_lib/registration/cost_functions/landmarks.h>
#include <deform_lib/registration/cost_functions/regularizer.h>
#include <deform_lib/registration/cost_functions/soft_constraints.h>
#include <deform_lib/registration/cost_functions/squared_distance.h>
#include <deform_lib/registration/cost_functions/unary_function.h>
#include <deform_lib/registration/displacement_field.h>
#include <deform_lib/registration/registration.h>
#include <deform_lib/registration/settings.h>
#include <deform_lib/registration/transform.h>
#include <deform_lib/registration/webgpu/block_grid.h>
#include <deform_lib/registration/webgpu/cost_functions.h>
#include <deform_lib/registration/webgpu/displacement_field.h>

#include <stk/image/volume.h>
#include <stk/image/webgpu_volume.h>
#include <stk/webgpu/webgpu.h>

#include <cmath>
#include <vector>

using Catch::Detail::Approx;

namespace {
    stk::VolumeFloat4 volume_float3_to_float4(const stk::VolumeFloat3& df)
    {
        dim3 dims = df.size();
        stk::VolumeFloat4 out(dims);
        out.copy_meta_from(df);

        for (int3 p : dims) {
            out(p) = float4{df(p).x, df(p).y, df(p).z, 0.0f};
        }
        return out;
    }

    // Cost buffers where every block in the grid is assigned a slot, in a
    // scrambled order.
    webgpu::CostBuffers make_cost_buffers(const webgpu::BlockGrid& grid,
                                          std::vector<uint32_t>& slots)
    {
        const int n = grid.num_blocks();
        slots.resize(n);
        for (int i = 0; i < n; ++i) {
            slots[i] = uint32_t((i * 7 + 3) % n);
        }

        webgpu::CostBuffers cost;
        cost.grid = grid;
        cost.slots = stk::webgpu::Buffer(n * sizeof(uint32_t));
        cost.slots.upload(slots.data(), n * sizeof(uint32_t));
        cost.unary = stk::webgpu::Buffer(n * grid.unary_slot_size() * sizeof(float2));
        for (int i = 0; i < 3; ++i) {
            cost.binary[i] = stk::webgpu::Buffer(
                n * grid.binary_slot_size(i) * sizeof(float4));
        }
        return cost;
    }

    template<typename T>
    std::vector<T> download(const stk::webgpu::Buffer& buffer)
    {
        std::vector<T> out(buffer.size() / sizeof(T));
        buffer.download(out.data(), out.size() * sizeof(T));
        return out;
    }

    // A shifted grid of blocks that does not divide the volume evenly
    webgpu::BlockGrid make_grid(const dim3& dims)
    {
        return webgpu::BlockGrid{
            int3{int(dims.x), int(dims.y), int(dims.z)},
            int3{8, 8, 8},
            int3{4, 4, 4},
            int3{
                (int(dims.x) + 4 + 7) / 8,
                (int(dims.y) + 4 + 7) / 8,
                (int(dims.z) + 4 + 7) / 8
            }
        };
    }

    int3 block_of(const webgpu::BlockGrid& grid, const int3& p)
    {
        return int3{
            (p.x + grid.block_offset.x) / grid.block_dims.x,
            (p.y + grid.block_offset.y) / grid.block_dims.y,
            (p.z + grid.block_offset.z) / grid.block_dims.z
        };
    }

    bool skip()
    {
        if (!stk::webgpu::available()) {
            WARN("No WebGPU device available, skipping");
            return true;
        }
        return false;
    }
}

TEST_CASE("webgpu_gaussian_filter", "[webgpu]")
{
    if (skip()) return;

    stk::VolumeFloat src({16,17,18});
    src.set_spacing({1.0f, 1.5f, 2.0f});

    for (int3 p : src.size()) {
        int r = (p.x-7)*(p.x-7) + (p.y-7)*(p.y-7) + (p.z-7)*(p.z-7);
        src(p) = r < 8*8 ? 1.0f : 0.0f;
    }

    // Use CPU-version as ground truth
    stk::VolumeFloat out = filters::gaussian_filter_3d(src, 2.5f);

    stk::WebGpuVolume gpu_src(src);
    stk::VolumeFloat gpu_out = filters::webgpu::gaussian_filter_3d(gpu_src, 2.5f).download();

    CHECK(gpu_out.spacing().y == Approx(out.spacing().y));
    for (int3 p : src.size()) {
        REQUIRE(gpu_out(p) == Approx(out(p)).margin(1e-6));
    }
}

TEST_CASE("webgpu_downsample_gaussian", "[webgpu]")
{
    if (skip()) return;

    stk::VolumeFloat src({16,15,14});
    for (int3 p : src.size()) {
        int r = (p.x-7)*(p.x-7) + (p.y-7)*(p.y-7) + (p.z-7)*(p.z-7);
        src(p) = r < 5*5 ? 1.0f : 0.0f;
    }

    stk::VolumeFloat out = filters::downsample_volume_by_2(src);
    stk::VolumeFloat gpu_out = filters::webgpu::downsample_volume_by_2(
        stk::WebGpuVolume(src)).download();

    REQUIRE(gpu_out.size() == out.size());
    CHECK(gpu_out.spacing().x == Approx(out.spacing().x));
    CHECK(gpu_out.spacing().y == Approx(out.spacing().y));
    CHECK(gpu_out.spacing().z == Approx(out.spacing().z));

    for (int3 p : out.size()) {
        REQUIRE(gpu_out(p) == Approx(out(p)).margin(1e-6));
    }
}

TEST_CASE("webgpu_resample_vectorfield", "[webgpu]")
{
    if (skip()) return;

    stk::VolumeFloat3 src({8,9,10});
    for (int3 p : src.size()) {
        src(p) = float3{float(p.x), 0.5f * p.y, std::sin(float(p.z))};
    }
    stk::WebGpuVolume gpu_src(volume_float3_to_float4(src));

    SECTION("upsample")
    {
        dim3 new_dims{16, 17, 20};
        stk::VolumeFloat3 out = filters::upsample_vectorfield(src, new_dims);
        stk::VolumeFloat4 gpu_out = filters::webgpu::upsample_vectorfield(
            gpu_src, new_dims).download();

        REQUIRE(gpu_out.size() == out.size());
        CHECK(gpu_out.spacing().y == Approx(out.spacing().y));
        for (int3 p : out.size()) {
            REQUIRE(gpu_out(p).x == Approx(out(p).x).margin(1e-5));
            REQUIRE(gpu_out(p).y == Approx(out(p).y).margin(1e-5));
            REQUIRE(gpu_out(p).z == Approx(out(p).z).margin(1e-5));
        }
    }
    SECTION("downsample")
    {
        stk::VolumeFloat3 out = filters::downsample_vectorfield_by_2(src);
        stk::VolumeFloat4 gpu_out = filters::webgpu::downsample_vectorfield_by_2(
            gpu_src).download();

        REQUIRE(gpu_out.size() == out.size());
        for (int3 p : out.size()) {
            REQUIRE(gpu_out(p).x == Approx(out(p).x).margin(1e-5));
            REQUIRE(gpu_out(p).y == Approx(out(p).y).margin(1e-5));
            REQUIRE(gpu_out(p).z == Approx(out(p).z).margin(1e-5));
        }
    }
}

TEST_CASE("webgpu_binary_term", "[webgpu]")
{
    if (skip()) return;

    float3 delta {0.5f, 0.0f, 0.0f};
    dim3 dims { 20, 19, 18 };
    stk::VolumeFloat3 vectorfield(dims, float3{0, 0, 0});
    for (int3 p : dims) {
        vectorfield(p) = float3{
            (rand() % 100) / 10.0f,
            (rand() % 100) / 10.0f,
            (rand() % 100) / 10.0f
        };
    }
    stk::WebGpuVolume gpu_vectorfield(volume_float3_to_float4(vectorfield));

    Matrix3x3f matrix {
        float3{2, 0, 0},
        float3{0, 3, 0},
        float3{0, 0, 4}
    };

    AffineTransform affine_transform(matrix, float3{5,-6,7});

    Settings::UpdateRule update_rules[] = {
        Settings::UpdateRule_Additive,
        Settings::UpdateRule_Compositive
    };
    for (auto update_rule : update_rules) {
        webgpu::BlockGrid grid = make_grid(dims);
        std::vector<uint32_t> slots;
        webgpu::CostBuffers cost = make_cost_buffers(grid, slots);

        WebGpuBinaryFunction gpu_binary_fn;
        gpu_binary_fn.set_fixed_spacing(float3{1, 1, 1});
        gpu_binary_fn.set_regularization_weight(0.25f);
        gpu_binary_fn.set_regularization_scale(1.0f);
        gpu_binary_fn.set_regularization_exponent(2.0f);

        gpu_binary_fn(
            WebGpuDisplacementField(gpu_vectorfield, affine_transform),
            delta,
            update_rule,
            cost
        );

        std::vector<float4> binary_cost[3] = {
            download<float4>(cost.binary[0]),
            download<float4>(cost.binary[1]),
            download<float4>(cost.binary[2])
        };
        stk::webgpu::check_errors();

        DisplacementField df(vectorfield, affine_transform);

        Regularizer binary_fn;
        binary_fn.set_fixed_spacing(float3{1, 1, 1});
        binary_fn.set_regularization_weight(0.25f);
        binary_fn.set_regularization_scale(1.0f);
        binary_fn.set_regularization_exponent(2.0f);

        const bool composite = update_rule == Settings::UpdateRule_Compositive;
        const int3 steps[3] = {int3{1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};

        for (int3 p : dims) {
            float3 d1 = df.get(p);
            float3 d1d = df.get(p, delta, composite);

            const int3 blk = block_of(grid, p);
            const uint32_t slot = slots[grid.block_index(blk)];
            const int3 l = p - grid.block_begin(blk);

            for (int axis = 0; axis < 3; ++axis) {
                const int3 step = steps[axis];
                const int3 q = p + step;
                if (q.x >= int(dims.x) || q.y >= int(dims.y) || q.z >= int(dims.z))
                    continue;

                float3 d2 = df.get(q);
                float3 d2d = df.get(q, delta, composite);

                double f00 = binary_fn(p, d1, d2, step);
                double f01 = binary_fn(p, d1, d2d, step);
                double f10 = binary_fn(p, d1d, d2, step);
                double f11 = binary_fn(p, d1d, d2d, step);

                const float4 c = binary_cost[axis][grid.binary_index(axis, slot, l)];
                REQUIRE(c.x == Approx(f00));
                REQUIRE(c.y == Approx(f01));
                REQUIRE(c.z == Approx(f10));
                REQUIRE(c.w == Approx(f11));

                // The cost should also be available in the padding of the
                // neighbouring block
                const int3 q_blk = block_of(grid, q);
                if (grid.block_index(q_blk) != grid.block_index(blk)) {
                    const uint32_t q_slot = slots[grid.block_index(q_blk)];
                    const int3 q_l = q - grid.block_begin(q_blk) - step;

                    const float4 c2 = binary_cost[axis][grid.binary_index(axis, q_slot, q_l)];
                    REQUIRE(c2.x == Approx(f00));
                    REQUIRE(c2.y == Approx(f01));
                }
            }
        }
    } // for update_rule
}

TEST_CASE("webgpu_unary_term", "[webgpu]")
{
    if (skip()) return;

    float3 delta {0.5f, 0.0f, 0.0f};
    dim3 dims { 20, 19, 18 };
    stk::VolumeFloat3 vectorfield(dims, float3{0, 0, 0});
    stk::VolumeFloat fixed(dims, 0.0f);
    stk::VolumeFloat moving(dims, 0.0f);
    for (int3 p : dims) {
        vectorfield(p) = float3{
            (rand() % 100) / 10.0f,
            (rand() % 100) / 10.0f,
            (rand() % 100) / 10.0f
        };
        fixed(p) = (rand() % 100) / 100.0f;
        moving(p) = (rand() % 100) / 100.0f;
    }
    moving.set_origin({1.0f, -2.0f, 0.5f});
    moving.set_spacing({1.1f, 0.9f, 1.2f});

    stk::WebGpuVolume gpu_vectorfield(volume_float3_to_float4(vectorfield));
    stk::WebGpuVolume gpu_fixed(fixed);
    stk::WebGpuVolume gpu_moving(moving);

    Matrix3x3f matrix {
        float3{1.1f, 0, 0},
        float3{0, 0.9f, 0},
        float3{0, 0, 1.2f}
    };

    AffineTransform affine_transform(matrix, float3{2,-1,1});
    DisplacementField df(vectorfield, affine_transform);

    Settings::UpdateRule update_rules[] = {
        Settings::UpdateRule_Additive,
        Settings::UpdateRule_Compositive
    };

    for (int function = 0; function < 2; ++function) {
    for (auto update_rule : update_rules) {
        webgpu::BlockGrid grid = make_grid(dims);
        std::vector<uint32_t> slots;
        webgpu::CostBuffers cost = make_cost_buffers(grid, slots);

        WebGpuUnaryFunction gpu_unary_fn;
        UnaryFunction unary_fn;

        std::unique_ptr<WebGpuCostFunction> fn;
        if (function == 0) {
            fn = make_unique<WebGpuCostFunction_NCC>(gpu_fixed, gpu_moving, 2);
            unary_fn.add_function(
                make_unique<NCCFunction_sphere<float>>(fixed, moving, 2), 0.7f);
        }
        else {
            fn = make_unique<WebGpuCostFunction_SSD>(gpu_fixed, gpu_moving);
            unary_fn.add_function(
                make_unique<SquaredDistanceFunction<float>>(fixed, moving), 0.7f);
        }
        gpu_unary_fn.add_function(fn, 0.7f);

        WebGpuDisplacementField gpu_df(gpu_vectorfield, affine_transform);
        gpu_unary_fn(gpu_df, delta, update_rule, cost);
        // Costs should accumulate
        gpu_unary_fn(gpu_df, delta, update_rule, cost);

        std::vector<float2> unary_cost = download<float2>(cost.unary);
        stk::webgpu::check_errors();

        for (int3 p : dims) {
            float3 d[2] = {
                df.get(p),
                df.get(p, delta, update_rule == Settings::UpdateRule_Compositive)
            };

            const int3 blk = block_of(grid, p);
            const uint32_t slot = slots[grid.block_index(blk)];
            const float2 c = unary_cost[grid.unary_index(slot, p - grid.block_begin(blk))];
            const float gpu_f[2] = {c.x, c.y};

            for (int i = 0; i < 2; ++i) {
                // The CPU version of SSD sets the cost to zero outside the
                // moving volume, while the GPU versions sample zeros.
                const float3 moving_p = moving.point2index(fixed.index2point(p) + d[i]);
                if (function == 1 && (
                    moving_p.x < 0 || moving_p.x >= dims.x ||
                    moving_p.y < 0 || moving_p.y >= dims.y ||
                    moving_p.z < 0 || moving_p.z >= dims.z))
                    continue;

                REQUIRE(gpu_f[i] == Approx(2.0 * unary_fn(p, d[i])).margin(1e-4));
            }
        }
    }
    }
}

TEST_CASE("webgpu_unary_term_other", "[webgpu]")
{
    if (skip()) return;

    float3 delta {0.0f, 0.5f, 0.0f};
    dim3 dims { 20, 19, 18 };
    stk::VolumeFloat3 vectorfield(dims, float3{0, 0, 0});
    stk::VolumeFloat fixed(dims, 0.0f);
    stk::VolumeFloat moving(dims, 0.0f);
    stk::VolumeFloat fixed_mask(dims, 0.0f);
    stk::VolumeFloat moving_mask(dims, 0.0f);
    stk::VolumeUChar constraint_mask(dims, uint8_t{0});
    stk::VolumeFloat constraint_mask_float(dims, 0.0f);
    stk::VolumeFloat3 constraint_values(dims, float3{0, 0, 0});
    for (int3 p : dims) {
        // Keep the points within the moving volume, see webgpu_unary_term
        vectorfield(p) = float3{
            (rand() % 100) / 100.0f,
            (rand() % 100) / 100.0f,
            (rand() % 100) / 100.0f
        };
        fixed(p) = (rand() % 100) / 100.0f;
        moving(p) = (rand() % 100) / 100.0f;
        fixed_mask(p) = (rand() % 3) / 2.0f;
        moving_mask(p) = (rand() % 3) / 2.0f;
        constraint_mask(p) = uint8_t(rand() % 2);
        constraint_mask_float(p) = float(constraint_mask(p));
        constraint_values(p) = float3{
            (rand() % 100) / 50.0f,
            (rand() % 100) / 50.0f,
            (rand() % 100) / 50.0f
        };
    }
    fixed.set_origin({3.0f, 2.0f, 1.0f});
    fixed.set_spacing({1.5f, 1.25f, 2.0f});
    vectorfield.copy_meta_from(fixed);

    stk::WebGpuVolume gpu_vectorfield(volume_float3_to_float4(vectorfield));
    stk::WebGpuVolume gpu_fixed(fixed);
    stk::WebGpuVolume gpu_moving(moving);

    std::vector<float3> fixed_landmarks{{4, 5, 6}, {10.5f, 11, 12}, {20, 3, 9}};
    std::vector<float3> moving_landmarks{{5, 5, 7}, {10, 12, 12}, {19, 4, 8}};

    DisplacementField df(vectorfield);
    WebGpuDisplacementField gpu_df(gpu_vectorfield);

    for (int function = 0; function < 3; ++function) {
        webgpu::BlockGrid grid = make_grid(dims);
        std::vector<uint32_t> slots;
        webgpu::CostBuffers cost = make_cost_buffers(grid, slots);

        WebGpuUnaryFunction gpu_unary_fn;
        UnaryFunction unary_fn;

        std::unique_ptr<WebGpuCostFunction> fn;
        if (function == 0) {
            fn = make_unique<WebGpuCostFunction_Landmarks>(
                fixed_landmarks, moving_landmarks, 2.0f);
            unary_fn.add_function(make_unique<LandmarksFunction>(
                fixed_landmarks, moving_landmarks, fixed, 2.0f), 0.5f);
        }
        else if (function == 1) {
            fn = make_unique<WebGpuCostFunction_SoftConstraints>(
                stk::WebGpuVolume(constraint_mask_float),
                stk::WebGpuVolume(volume_float3_to_float4(constraint_values)));
            unary_fn.add_function(make_unique<SoftConstraintsFunction>(
                constraint_mask, constraint_values), 0.5f);
        }
        else {
            // Masks
            fn = make_unique<WebGpuCostFunction_SSD>(gpu_fixed, gpu_moving);
            fn->set_fixed_mask(stk::WebGpuVolume(fixed_mask));
            fn->set_moving_mask(stk::WebGpuVolume(moving_mask));

            auto cpu_fn = make_unique<SquaredDistanceFunction<float>>(fixed, moving);
            cpu_fn->set_moving_mask(moving_mask);
            unary_fn.add_function(std::move(cpu_fn), 0.5f);
            unary_fn.set_fixed_mask(fixed_mask);
        }
        gpu_unary_fn.add_function(fn, 0.5f);
        gpu_unary_fn(gpu_df, delta, Settings::UpdateRule_Additive, cost);

        std::vector<float2> unary_cost = download<float2>(cost.unary);
        stk::webgpu::check_errors();

        for (int3 p : dims) {
            const int3 blk = block_of(grid, p);
            const uint32_t slot = slots[grid.block_index(blk)];
            const float2 c = unary_cost[grid.unary_index(slot, p - grid.block_begin(blk))];

            REQUIRE(c.x == Approx(unary_fn(p, df.get(p))).margin(1e-5));
            REQUIRE(c.y == Approx(unary_fn(p, df.get(p, delta, false))).margin(1e-5));
        }
    }
}

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
TEST_CASE("webgpu_binary_term_weight_map", "[webgpu]")
{
    if (skip()) return;

    float3 delta {0.0f, 0.0f, 0.5f};
    dim3 dims { 20, 19, 18 };
    stk::VolumeFloat3 vectorfield(dims, float3{0, 0, 0});
    stk::VolumeFloat weight_map(dims, 0.0f);
    for (int3 p : dims) {
        vectorfield(p) = float3{
            (rand() % 100) / 10.0f,
            (rand() % 100) / 10.0f,
            (rand() % 100) / 10.0f
        };
        weight_map(p) = (rand() % 100) / 100.0f;
    }
    vectorfield.set_spacing({1.5f, 1.25f, 2.0f});

    webgpu::BlockGrid grid = make_grid(dims);
    std::vector<uint32_t> slots;
    webgpu::CostBuffers cost = make_cost_buffers(grid, slots);

    WebGpuBinaryFunction gpu_binary_fn;
    Regularizer binary_fn;

    gpu_binary_fn.set_fixed_spacing(vectorfield.spacing());
    gpu_binary_fn.set_regularization_scale(0.5f);
    gpu_binary_fn.set_regularization_exponent(1.5f);
    gpu_binary_fn.set_weight_map(stk::WebGpuVolume(weight_map));

    binary_fn.set_fixed_spacing(vectorfield.spacing());
    binary_fn.set_regularization_scale(0.5f);
    binary_fn.set_regularization_exponent(1.5f);
    binary_fn.set_weight_map(weight_map);

    gpu_binary_fn(
        WebGpuDisplacementField(stk::WebGpuVolume(volume_float3_to_float4(vectorfield))),
        delta,
        Settings::UpdateRule_Additive,
        cost
    );

    std::vector<float4> binary_cost[3] = {
        download<float4>(cost.binary[0]),
        download<float4>(cost.binary[1]),
        download<float4>(cost.binary[2])
    };
    stk::webgpu::check_errors();

    DisplacementField df(vectorfield);
    const int3 steps[3] = {int3{1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};

    for (int3 p : dims) {
        const int3 blk = block_of(grid, p);
        const uint32_t slot = slots[grid.block_index(blk)];
        const int3 l = p - grid.block_begin(blk);

        for (int axis = 0; axis < 3; ++axis) {
            const int3 q = p + steps[axis];
            if (q.x >= int(dims.x) || q.y >= int(dims.y) || q.z >= int(dims.z))
                continue;

            const float4 c = binary_cost[axis][grid.binary_index(axis, slot, l)];
            REQUIRE(c.x == Approx(binary_fn(p, df.get(p), df.get(q), steps[axis])));
            REQUIRE(c.w == Approx(binary_fn(
                p, df.get(p, delta, false), df.get(q, delta, false), steps[axis])));
        }
    }
}
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

namespace {
    // A blob, to be registered to a translated version of itself
    stk::VolumeFloat make_blob(const dim3& dims, const float3& center)
    {
        stk::VolumeFloat vol(dims);
        for (int3 p : dims) {
            const float dx = (p.x - center.x) / 6.0f;
            const float dy = (p.y - center.y) / 6.0f;
            const float dz = (p.z - center.z) / 6.0f;
            vol(p) = std::exp(-(dx*dx + dy*dy + dz*dz));
        }
        return vol;
    }

    double mean_squared_difference(const stk::VolumeFloat& a, const stk::VolumeFloat& b)
    {
        double sum = 0;
        for (int3 p : a.size()) {
            sum += double(a(p) - b(p)) * double(a(p) - b(p));
        }
        return sum / double(a.size().x * a.size().y * a.size().z);
    }

    stk::VolumeFloat3 run_registration(
        const char* settings_str,
        const stk::VolumeFloat& fixed,
        const stk::VolumeFloat& moving,
        bool use_gpu)
    {
        Settings settings;
        parse_registration_settings(settings_str, settings);

        std::vector<stk::Volume> fixed_volumes{fixed};
        std::vector<stk::Volume> moving_volumes{moving};

        return registration(
            settings,
            fixed_volumes,
            moving_volumes,
            stk::Volume(),
            stk::Volume(),
            {},
            {},
            stk::Volume(),
            AffineTransform(),
            stk::Volume(),
            stk::Volume(),
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
            stk::Volume(),
#endif
            0,
            use_gpu
        );
    }
}

TEST_CASE("webgpu_registration", "[webgpu]")
{
    if (skip()) return;

    // Sizes that are not multiples of the block size
    dim3 dims{40, 36, 30};
    stk::VolumeFloat fixed = make_blob(dims, float3{20, 18, 15});
    stk::VolumeFloat moving = make_blob(dims, float3{23, 16, 16});

    const char* settings[] = {
        // Default settings
        "pyramid_levels: 3\n"
        "image_slots:\n"
        "  - cost_function: ssd\n"
        "    normalize: false\n",

        "pyramid_levels: 3\n"
        "update_rule: compositive\n"
        "image_slots:\n"
        "  - cost_function: ncc\n"
        "    normalize: false\n"
    };

    const double initial = mean_squared_difference(fixed, moving);

    for (const char* s : settings) {
        stk::VolumeFloat3 cpu_df = run_registration(s, fixed, moving, false);
        stk::VolumeFloat3 gpu_df = run_registration(s, fixed, moving, true);
        REQUIRE(gpu_df.valid());
        REQUIRE(gpu_df.size() == dims);

        stk::VolumeFloat cpu_result = transform_volume(moving, cpu_df);
        stk::VolumeFloat gpu_result = transform_volume(moving, gpu_df);

        const double cpu_msd = mean_squared_difference(fixed, cpu_result);
        const double gpu_msd = mean_squared_difference(fixed, gpu_result);
        INFO("Initial: " << initial << ", CPU: " << cpu_msd << ", WebGPU: " << gpu_msd);

        // The engines are not expected to give identical results, but the
        // quality of the registration should be comparable.
        CHECK(gpu_msd < 0.1 * initial);
        CHECK(gpu_msd < 2.0 * cpu_msd + 1e-6);
    }
}

#endif // DF_USE_WEBGPU
