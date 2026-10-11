#include "displacement_field.h"

#ifdef DF_USE_WEBGPU

#include <stk/common/assert.h>

WebGpuDisplacementField::WebGpuDisplacementField()
{
}
WebGpuDisplacementField::WebGpuDisplacementField(
    const stk::WebGpuVolume& df,
    const AffineTransform& affine_transform
) :
    _df(df), _affine_transform(affine_transform)
{
    ASSERT(_df.voxel_type() == stk::Type_Float4);
}
WebGpuDisplacementField::~WebGpuDisplacementField()
{
}
WebGpuDisplacementField WebGpuDisplacementField::clone() const
{
    return WebGpuDisplacementField(_df.clone(), _affine_transform);
}
void WebGpuDisplacementField::copy_from(const WebGpuDisplacementField& other)
{
    _df.copy_from(other._df);
    _affine_transform = other._affine_transform;
}
dim3 WebGpuDisplacementField::size() const
{
    return _df.size();
}
const stk::WebGpuVolume& WebGpuDisplacementField::volume() const
{
    return _df;
}
const AffineTransform& WebGpuDisplacementField::affine_transform() const
{
    return _affine_transform;
}
const float3& WebGpuDisplacementField::origin() const
{
    return _df.origin();
}
const float3& WebGpuDisplacementField::spacing() const
{
    return _df.spacing();
}
const Matrix3x3f& WebGpuDisplacementField::direction() const
{
    return _df.direction();
}
webgpu::DisplacementFieldParams WebGpuDisplacementField::params() const
{
    const float3 o = _df.origin();
    const float3 s = _df.spacing();
    const float3 b = _affine_transform.offset();
    const dim3 dims = _df.size();

    webgpu::DisplacementFieldParams p = {};
    const float origin[3] = {o.x, o.y, o.z};
    const float spacing[3] = {s.x, s.y, s.z};
    const float offset[3] = {b.x, b.y, b.z};
    for (int i = 0; i < 3; ++i) {
        p.origin[i] = origin[i];
        p.spacing[i] = spacing[i];
        p.inv_spacing[i] = 1.0f / spacing[i];
        p.affine_offset[i] = offset[i];
    }
    webgpu::set_rows(p.direction, _df.direction());
    webgpu::set_rows(p.affine_matrix, _affine_transform.matrix());
    p.dims[0] = int(dims.x);
    p.dims[1] = int(dims.y);
    p.dims[2] = int(dims.z);
    return p;
}

stk::WebGpuVolume webgpu::compute_displacement_field(
    const stk::WebGpuVolume& vector_field,
    const AffineTransform& affine
)
{
    ASSERT(vector_field.voxel_type() == stk::Type_Float4);

    static stk::webgpu::Kernel kernel(
        compute_displacement_field_shader(),
        {stk::webgpu::Access_Read, stk::webgpu::Access_ReadWrite},
        sizeof(DisplacementFieldParams)
    );

    stk::WebGpuVolume out(vector_field.size(), stk::Type_Float4);
    out.copy_meta_from(vector_field);

    DisplacementFieldParams params = WebGpuDisplacementField(vector_field, affine).params();
    kernel.run(&params, {vector_field.buffer(), out.buffer()}, vector_field.size());
    return out;
}

void webgpu::set_rows(float dst[3][4], const Matrix3x3f& m)
{
    for (int i = 0; i < 3; ++i) {
        dst[i][0] = m._rows[i].x;
        dst[i][1] = m._rows[i].y;
        dst[i][2] = m._rows[i].z;
        dst[i][3] = 0.0f;
    }
}

#endif // DF_USE_WEBGPU
