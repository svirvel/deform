#pragma once

#include <deform_lib/config.h>

#ifdef DF_USE_WEBGPU

#include "shaders.h"
#include "../affine_transform.h"

#include <stk/image/webgpu_volume.h>

/** Wrapper for a displacement field image on a WebGPU device.
 * WebGPU version of GpuDisplacementField.
 * */
class WebGpuDisplacementField
{
public:
    WebGpuDisplacementField();
    WebGpuDisplacementField(
        const stk::WebGpuVolume& df,
        const AffineTransform& affine_transform = AffineTransform()
    );
    ~WebGpuDisplacementField();

    // Clones the displacement field and underlying data
    WebGpuDisplacementField clone() const;

    // Copies another displacement field into this
    void copy_from(const WebGpuDisplacementField& other);

    // Returns the dimensions of the displacement field
    dim3 size() const;

    // Volume containing the displacements only
    const stk::WebGpuVolume& volume() const;
    const AffineTransform& affine_transform() const;

    const float3& origin() const;
    const float3& spacing() const;
    const Matrix3x3f& direction() const;

    // Parameters for the kernels
    webgpu::DisplacementFieldParams params() const;

private:
    stk::WebGpuVolume _df;
    AffineTransform _affine_transform;
};

namespace webgpu {
    // Computes the composite of a vector field and an affine transform:
    // u'(x) <- A(x + u(x)) + b - x
    stk::WebGpuVolume compute_displacement_field(
        const stk::WebGpuVolume& vector_field,
        const AffineTransform& affine
    );

    // Copies a matrix to rows in a uniform block
    void set_rows(float dst[3][4], const Matrix3x3f& m);
}

#endif // DF_USE_WEBGPU
