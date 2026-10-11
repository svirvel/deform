#include "filters.h"

#ifdef DF_USE_WEBGPU

#include <deform_lib/registration/webgpu/shaders.h>

#include <stk/common/assert.h>
#include <stk/image/webgpu_volume.h>
#include <stk/math/float3.h>
#include <stk/webgpu/webgpu.h>

#include <algorithm>
#include <cmath>

namespace {
    using stk::webgpu::Access_Read;
    using stk::webgpu::Access_ReadWrite;

    const char* wgsl_type(stk::Type type)
    {
        return type == stk::Type_Float ? "f32" : "vec4<f32>";
    }

    void set_dims(int32_t dst[4], const dim3& dims)
    {
        dst[0] = int32_t(dims.x);
        dst[1] = int32_t(dims.y);
        dst[2] = int32_t(dims.z);
        dst[3] = 0;
    }

    /// Shrinks the volume by removing every other element
    stk::WebGpuVolume shrink_volume_by_2(const stk::WebGpuVolume& src)
    {
        ASSERT(src.voxel_type() == stk::Type_Float ||
               src.voxel_type() == stk::Type_Float4);

        static stk::webgpu::Kernel kernel_float(
            webgpu::shrink_by_2_shader("f32"),
            {Access_Read, Access_ReadWrite}, sizeof(webgpu::FilterParams));
        static stk::webgpu::Kernel kernel_float4(
            webgpu::shrink_by_2_shader("vec4<f32>"),
            {Access_Read, Access_ReadWrite}, sizeof(webgpu::FilterParams));

        dim3 old_dims = src.size();
        dim3 new_dims {
            uint32_t(ceil(old_dims.x * 0.5f)),
            uint32_t(ceil(old_dims.y * 0.5f)),
            uint32_t(ceil(old_dims.z * 0.5f)),
        };

        stk::WebGpuVolume dest(new_dims, src.voxel_type());
        dest.copy_meta_from(src);

        float3 old_spacing = src.spacing();
        float3 new_spacing {
            old_spacing.x * (old_dims.x / float(new_dims.x)),
            old_spacing.y * (old_dims.y / float(new_dims.y)),
            old_spacing.z * (old_dims.z / float(new_dims.z))
        };
        dest.set_spacing(new_spacing);

        webgpu::FilterParams params = {};
        set_dims(params.src_dims, old_dims);
        set_dims(params.dst_dims, new_dims);

        auto& kernel = src.voxel_type() == stk::Type_Float ? kernel_float : kernel_float4;
        kernel.run(&params, {src.buffer(), dest.buffer()}, new_dims);

        return dest;
    }
}

namespace filters
{
namespace webgpu
{
    stk::WebGpuVolume gaussian_filter_3d(const stk::WebGpuVolume& volume, float sigma)
    {
        ASSERT(volume.valid());
        ASSERT(volume.voxel_type() == stk::Type_Float ||
               volume.voxel_type() == stk::Type_Float4);

        static stk::webgpu::Kernel kernel_float(
            ::webgpu::gaussian_filter_shader("f32"),
            {Access_Read, Access_ReadWrite}, sizeof(::webgpu::FilterParams));
        static stk::webgpu::Kernel kernel_float4(
            ::webgpu::gaussian_filter_shader("vec4<f32>"),
            {Access_Read, Access_ReadWrite}, sizeof(::webgpu::FilterParams));

        auto& kernel = volume.voxel_type() == stk::Type_Float ? kernel_float : kernel_float4;

        dim3 dims = volume.size();
        float3 spacing = volume.spacing();

        stk::WebGpuVolume tmp(dims, volume.voxel_type());
        stk::WebGpuVolume out(dims, volume.voxel_type());
        out.copy_meta_from(volume);

        ::webgpu::FilterParams params = {};
        set_dims(params.src_dims, dims);
        set_dims(params.dst_dims, dims);
        params.fparams[0] = -1.0f / (2.0f * sigma * sigma);

        // x: volume -> out
        params.fparams[1] = spacing.x;
        params.iparams[0] = (int)ceilf(3 * sigma / spacing.x);
        params.iparams[1] = 0;
        kernel.run(&params, {volume.buffer(), out.buffer()}, dims);

        // y: out -> tmp
        params.fparams[1] = spacing.y;
        params.iparams[0] = (int)ceilf(3 * sigma / spacing.y);
        params.iparams[1] = 1;
        kernel.run(&params, {out.buffer(), tmp.buffer()}, dims);

        // z: tmp -> out
        params.fparams[1] = spacing.z;
        params.iparams[0] = (int)ceilf(3 * sigma / spacing.z);
        params.iparams[1] = 2;
        kernel.run(&params, {tmp.buffer(), out.buffer()}, dims);

        return out;
    }

    stk::WebGpuVolume downsample_volume_by_2(const stk::WebGpuVolume& vol)
    {
        ASSERT(vol.voxel_type() == stk::Type_Float);

        float3 spacing = vol.spacing();
        float unit_sigma = std::min(spacing.x, std::min(spacing.y, spacing.z));

        stk::WebGpuVolume filtered = gaussian_filter_3d(vol, unit_sigma);

        return shrink_volume_by_2(filtered);
    }

    stk::WebGpuVolume downsample_vectorfield_by_2(const stk::WebGpuVolume& vol)
    {
        ASSERT(vol.voxel_type() == stk::Type_Float4);

        float3 spacing = vol.spacing();
        float unit_sigma = std::min(spacing.x, std::min(spacing.y, spacing.z));

        stk::WebGpuVolume filtered = gaussian_filter_3d(vol, unit_sigma);

        return shrink_volume_by_2(filtered);
    }

    stk::WebGpuVolume upsample_vectorfield(const stk::WebGpuVolume& vol, const dim3& new_dims)
    {
        ASSERT(vol.voxel_type() == stk::Type_Float4);

        static stk::webgpu::Kernel kernel(
            ::webgpu::upsample_vectorfield_shader(),
            {Access_Read, Access_ReadWrite}, sizeof(::webgpu::FilterParams));

        dim3 old_dims = vol.size();
        float3 inv_scale{
            float(old_dims.x) / new_dims.x,
            float(old_dims.y) / new_dims.y,
            float(old_dims.z) / new_dims.z
        };

        stk::WebGpuVolume out(new_dims, stk::Type_Float4);
        out.copy_meta_from(vol);

        float3 old_spacing = vol.spacing();
        float3 new_spacing{
            old_spacing.x * inv_scale.x,
            old_spacing.y * inv_scale.y,
            old_spacing.z * inv_scale.z
        };
        out.set_spacing(new_spacing);

        ::webgpu::FilterParams params = {};
        set_dims(params.src_dims, old_dims);
        set_dims(params.dst_dims, new_dims);
        params.fparams[0] = inv_scale.x;
        params.fparams[1] = inv_scale.y;
        params.fparams[2] = inv_scale.z;

        kernel.run(&params, {vol.buffer(), out.buffer()}, new_dims);

        return out;
    }
}
}

#endif // DF_USE_WEBGPU
