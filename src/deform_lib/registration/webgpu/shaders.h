#pragma once

#include <deform_lib/config.h>

#ifdef DF_USE_WEBGPU

#include <cstdint>
#include <string>

// WGSL sources for the WebGPU kernels, together with the uniform blocks they
// expect. The layout of the structs here have to match their WGSL
// counterparts, which is why they only consist of 16 byte vectors (and
// groups of four scalars).

namespace webgpu {

    // Regular grid of blocks. Costs are only computed for blocks that are
    // assigned a slot, see BlockGrid.
    struct GridParams
    {
        int32_t dims[4];
        int32_t block_dims[4];
        int32_t block_offset[4];
        int32_t block_count[4];
    };

    struct DisplacementFieldParams
    {
        float origin[4];
        float spacing[4];
        float inv_spacing[4];
        float direction[3][4]; // Rows
        float affine_matrix[3][4]; // Rows
        float affine_offset[4];
        int32_t dims[4];
    };

    struct MovingParams
    {
        int32_t dims[4];
        float origin[4];
        float inv_spacing[4];
        float inv_direction[3][4]; // Rows
    };

    struct UnaryParams
    {
        GridParams grid;
        DisplacementFieldParams df;
        MovingParams moving;
        float delta[4];
        float weight;
        int32_t update_rule;
        int32_t has_fixed_mask;
        int32_t has_moving_mask;
        // Specific for each cost function
        int32_t iparams[4];
        float fparams[4];
    };

    struct BinaryParams
    {
        GridParams grid;
        DisplacementFieldParams df;
        float delta[4];
        float inv_spacing2_exp[4];
        float weight;
        float scale;
        float half_exponent;
        int32_t update_rule;
        int32_t has_weight_map;
        int32_t _pad[3];
    };

    struct ApplyDeltaParams
    {
        DisplacementFieldParams df;
        float delta[4];
    };

    struct ReduceParams
    {
        GridParams grid;
        int32_t num_slots;
        int32_t _pad[3];
    };

    // Shared by the filters
    struct FilterParams
    {
        int32_t src_dims[4];
        int32_t dst_dims[4];
        float fparams[4];
        int32_t iparams[4];
    };

    enum CostFunction
    {
        CostFunction_SSD,
        CostFunction_NCC,
        CostFunction_Landmarks,
        CostFunction_SoftConstraints
    };

    // Unary cost functions.
    // Bindings: slots, df, input a, input b, fixed mask, moving mask, cost
    std::string unary_shader(CostFunction fn);

    // Regularization.
    // Bindings: slots, df, weight map, cost x, cost y, cost z
    std::string binary_shader();

    // Sums the energy for each z-layer of each slot.
    // Bindings: unary cost, cost x, cost y, cost z, sums
    std::string reduce_energy_shader();

    // Applies delta to all voxels with a non-zero label.
    // Bindings (additive): labels, df
    // Bindings (compositive): labels, df in, df out
    std::string apply_delta_shader(bool compositive);

    // u'(x) <- A(x + u(x)) + b - x
    // Bindings: df, out
    std::string compute_displacement_field_shader();

    // 1D gaussian filter, type is either "f32" or "vec4<f32>".
    // fparams: (factor, spacing), iparams: (filter size, axis)
    // Bindings: src, dst
    std::string gaussian_filter_shader(const std::string& type);

    // dst(p) = src(2*p)
    // Bindings: src, dst
    std::string shrink_by_2_shader(const std::string& type);

    // dst(p) = linear_at_clamp(src, p * inv_scale), fparams: inv_scale
    // Bindings: src, dst
    std::string upsample_vectorfield_shader();

    // Maximum of the 8 subvoxels.
    // Bindings: mask, dst
    std::string downsample_mask_shader();

    // Mean of the subvoxels within the mask.
    // Bindings: mask, values, dst
    std::string downsample_values_shader();

    // df(p) = values(p) where mask is set.
    // Bindings: mask, values, df
    std::string constrain_displacement_field_shader();
}

#endif // DF_USE_WEBGPU
