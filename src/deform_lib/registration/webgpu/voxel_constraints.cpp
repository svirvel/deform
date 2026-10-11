#include "voxel_constraints.h"

#ifdef DF_USE_WEBGPU

#include "displacement_field.h"
#include "shaders.h"

#include <stk/common/assert.h>
#include <stk/math/float3.h>

#include <cmath>

namespace {
    using stk::webgpu::Access_Read;
    using stk::webgpu::Access_ReadWrite;

    void set_dims(int32_t dst[4], const dim3& dims)
    {
        dst[0] = int32_t(dims.x);
        dst[1] = int32_t(dims.y);
        dst[2] = int32_t(dims.z);
        dst[3] = 0;
    }

    // Allocates the destination for a downsampling by 2
    stk::WebGpuVolume allocate_downsampled(const stk::WebGpuVolume& src)
    {
        dim3 old_dims = src.size();
        dim3 new_dims{
            uint32_t(ceil(old_dims.x * 0.5f)),
            uint32_t(ceil(old_dims.y * 0.5f)),
            uint32_t(ceil(old_dims.z * 0.5f)),
        };

        stk::WebGpuVolume result(new_dims, src.voxel_type());
        result.copy_meta_from(src);

        float3 old_spacing = src.spacing();
        float3 new_spacing {
            old_spacing.x * (old_dims.x / float(new_dims.x)),
            old_spacing.y * (old_dims.y / float(new_dims.y)),
            old_spacing.z * (old_dims.z / float(new_dims.z))
        };
        result.set_spacing(new_spacing);
        return result;
    }

    /*
        Downsampling of mask volumes by a factor of 2. Designed specifically for binary masks.
        Each resulting downsampled voxel takes the max value from the corresponding subvoxels
    */
    stk::WebGpuVolume downsample_mask_by_2(const stk::WebGpuVolume& mask)
    {
        static stk::webgpu::Kernel kernel(
            webgpu::downsample_mask_shader(),
            {Access_Read, Access_ReadWrite}, sizeof(webgpu::FilterParams));

        stk::WebGpuVolume result = allocate_downsampled(mask);

        webgpu::FilterParams params = {};
        set_dims(params.src_dims, mask.size());
        set_dims(params.dst_dims, result.size());
        kernel.run(&params, {mask.buffer(), result.buffer()}, result.size());

        return result;
    }

    /*
        Downsamples a constraint vector field.
        The value of each downsampled voxel is calculated as the mean of all subvoxels
            that are flagged as constraints (1s in the mask).
    */
    stk::WebGpuVolume downsample_values_by_2(
        const stk::WebGpuVolume& mask, const stk::WebGpuVolume& values)
    {
        static stk::webgpu::Kernel kernel(
            webgpu::downsample_values_shader(),
            {Access_Read, Access_Read, Access_ReadWrite}, sizeof(webgpu::FilterParams));

        stk::WebGpuVolume result = allocate_downsampled(values);

        webgpu::FilterParams params = {};
        set_dims(params.src_dims, values.size());
        set_dims(params.dst_dims, result.size());
        kernel.run(&params, {mask.buffer(), values.buffer(), result.buffer()}, result.size());

        return result;
    }
}

void webgpu_voxel_constraints::constraint_displacement_field(
    WebGpuDisplacementField& df,
    const stk::WebGpuVolume& mask,
    const stk::WebGpuVolume& values)
{
    if (mask.voxel_type() != stk::Type_Float
    || values.voxel_type() != stk::Type_Float4
    || mask.size() != values.size()) {
        FATAL() << "Invalid constraint volumes";
    }
    if (df.size() != values.size()) {
        FATAL() << "Invalid displacement field volume";
    }

    static stk::webgpu::Kernel kernel(
        webgpu::constrain_displacement_field_shader(),
        {Access_Read, Access_Read, Access_ReadWrite}, sizeof(webgpu::FilterParams));

    webgpu::FilterParams params = {};
    set_dims(params.src_dims, df.size());
    set_dims(params.dst_dims, df.size());
    kernel.run(&params, {mask.buffer(), values.buffer(), df.volume().buffer()}, df.size());
}

void webgpu_voxel_constraints::build_pyramids(
    const stk::WebGpuVolume& mask,
    const stk::WebGpuVolume& values,
    int num_levels,
    std::vector<stk::WebGpuVolume>& mask_pyramid,
    std::vector<stk::WebGpuVolume>& values_pyramid)
{
    if (mask.voxel_type() != stk::Type_Float
    || values.voxel_type() != stk::Type_Float4
    || mask.size() != values.size()) {
        FATAL() << "Invalid constraint volumes";
    }

    mask_pyramid.assign(num_levels, stk::WebGpuVolume());
    values_pyramid.assign(num_levels, stk::WebGpuVolume());

    mask_pyramid[0] = mask;
    values_pyramid[0] = values;

    for (int i = 0; i < num_levels-1; ++i) {
        mask_pyramid[i+1] = downsample_mask_by_2(mask_pyramid[i]);
        values_pyramid[i+1] = downsample_values_by_2(mask_pyramid[i], values_pyramid[i]);
    }
}

#endif // DF_USE_WEBGPU
