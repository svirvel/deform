#pragma once

#include <deform_lib/config.h>

#ifdef DF_USE_WEBGPU

#include <stk/image/dim3.h>

namespace stk
{
    class WebGpuVolume;
}

// WebGPU versions of the filters in filters/gpu.

namespace filters
{
namespace webgpu
{
    // Gaussian filter for float and float4 volumes
    stk::WebGpuVolume gaussian_filter_3d(const stk::WebGpuVolume& volume, float sigma);

    /// Applies a gaussian prefilter and downsamples the volume by 2.
    stk::WebGpuVolume downsample_volume_by_2(const stk::WebGpuVolume& vol);

    /// Downsamples a 3d vectorfield
    stk::WebGpuVolume downsample_vectorfield_by_2(const stk::WebGpuVolume& vol);

    /// Upsamples a 3d vectorfield
    stk::WebGpuVolume upsample_vectorfield(const stk::WebGpuVolume& vol, const dim3& new_dims);
}
}

#endif // DF_USE_WEBGPU
