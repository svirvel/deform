#pragma once

#include <deform_lib/config.h>

#ifdef DF_USE_WEBGPU

#include <stk/image/webgpu_volume.h>

#include <vector>

class WebGpuDisplacementField;

// WebGPU version of gpu_voxel_constraints. Note that the constraint mask is
// a float volume here, as WGSL has no 8-bit types.
namespace webgpu_voxel_constraints
{
    // Sets the displacement to the constraint value for all constrained voxels
    void constraint_displacement_field(
        WebGpuDisplacementField& df,
        const stk::WebGpuVolume& mask,
        const stk::WebGpuVolume& values
    );

    // mask     : float, non-zero for constrained voxels
    // values   : float4
    void build_pyramids(
        const stk::WebGpuVolume& mask,
        const stk::WebGpuVolume& values,
        int num_levels,
        std::vector<stk::WebGpuVolume>& mask_pyramid,
        std::vector<stk::WebGpuVolume>& values_pyramid
    );
}

#endif // DF_USE_WEBGPU
