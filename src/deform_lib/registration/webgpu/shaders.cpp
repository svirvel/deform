#include "shaders.h"

#ifdef DF_USE_WEBGPU

#include <stk/webgpu/webgpu.h>

namespace {

const char* MAIN_HEADER =
    "@compute @workgroup_size(" STK_WEBGPU_WORKGROUP_SIZE ")\n"
    "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n";

const char* COMMON = R"(
const FLT_EPSILON: f32 = 1.1920929e-7;
const NO_SLOT: u32 = 0xffffffffu;

fn vidx(p: vec3<i32>, dims: vec3<i32>) -> i32 {
    return p.x + dims.x * (p.y + dims.y * p.z);
}

fn mul3(m: array<vec4<f32>, 3>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(m[0].xyz, v), dot(m[1].xyz, v), dot(m[2].xyz, v));
}

fn norm2(v: vec3<f32>) -> f32 {
    return dot(v, v);
}
)";

// Requires `params.grid`
const char* GRID = R"(
struct Grid {
    dims: vec4<i32>,
    block_dims: vec4<i32>,
    block_offset: vec4<i32>,
    block_count: vec4<i32>,
};

fn block_of(p: vec3<i32>) -> vec3<i32> {
    return (p + params.grid.block_offset.xyz) / params.grid.block_dims.xyz;
}

// Coordinates of voxel within its block
fn block_local(p: vec3<i32>, blk: vec3<i32>) -> vec3<i32> {
    return p + params.grid.block_offset.xyz - blk * params.grid.block_dims.xyz;
}

fn unary_index(slot: u32, l: vec3<i32>) -> u32 {
    let b = vec3<u32>(params.grid.block_dims.xyz);
    let ul = vec3<u32>(l);
    return slot * (b.x * b.y * b.z) + ul.x + b.x * (ul.y + b.y * ul.z);
}

// The slots for the binary costs are padded by one in the negative direction
// of their axis, l is therefore allowed to be -1 for that axis.
fn binary_index(slot: u32, l: vec3<i32>, axis: vec3<i32>) -> u32 {
    let s = vec3<u32>(params.grid.block_dims.xyz + axis);
    let ul = vec3<u32>(l + axis);
    return slot * (s.x * s.y * s.z) + ul.x + s.x * (ul.y + s.y * ul.z);
}
)";

// Requires `params.grid` and the buffer `slots`
const char* GRID_SLOTS = R"(
fn block_slot(blk: vec3<i32>) -> u32 {
    let n = params.grid.block_count.xyz;
    return slots[blk.x + n.x * (blk.y + n.y * blk.z)];
}
)";

// Mirrors cuda::DisplacementField.
// Requires `params.df` and the buffer `df_vol`
const char* DISPLACEMENT_FIELD = R"(
struct Df {
    origin: vec4<f32>,
    spacing: vec4<f32>,
    inv_spacing: vec4<f32>,
    direction: array<vec4<f32>, 3>,
    affine_matrix: array<vec4<f32>, 3>,
    affine_offset: vec4<f32>,
    dims: vec4<i32>,
};

const UPDATE_RULE_ADDITIVE: i32 = 0;
const UPDATE_RULE_COMPOSITIVE: i32 = 1;

fn df_index2point(p: vec3<i32>) -> vec3<f32> {
    return params.df.origin.xyz
        + mul3(params.df.direction, params.df.spacing.xyz * vec3<f32>(p));
}

fn df_affine(pt: vec3<f32>) -> vec3<f32> {
    return mul3(params.df.affine_matrix, pt) + params.df.affine_offset.xyz;
}

// Displacement after applying delta according to the update rule
fn df_update(p: vec3<i32>, delta: vec3<f32>, update_rule: i32) -> vec3<f32> {
    let dims = params.df.dims.xyz;
    if (update_rule == UPDATE_RULE_COMPOSITIVE) {
        // Convert delta from mm to image space
        return df_linear_at_clamp(
            vec3<f32>(p) + delta * params.df.inv_spacing.xyz, dims
        ).xyz + delta;
    }
    return df_vol[vidx(p, dims)].xyz + delta;
}

fn df_get(p: vec3<i32>) -> vec3<f32> {
    let p1 = df_index2point(p);
    return df_affine(p1 + df_vol[vidx(p, params.df.dims.xyz)].xyz) - p1;
}

fn df_get_delta(p: vec3<i32>, delta: vec3<f32>, update_rule: i32) -> vec3<f32> {
    let p1 = df_index2point(p);
    return df_affine(p1 + df_update(p, delta, update_rule)) - p1;
}
)";

void replace_all(std::string& s, const std::string& from, const std::string& to)
{
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// Generates a function for trilinear interpolation in a buffer, mirrors
// stk::cuda::linear_at_border and stk::cuda::linear_at_clamp.
//  fn <name>(pos: vec3<f32>, dims: vec3<i32>) -> <type>
std::string linear_at(
    const std::string& name,
    const std::string& buffer,
    const std::string& type,
    bool clamp
)
{
    std::string s = R"(
fn NAME(pos: vec3<f32>, dims: vec3<i32>) -> TYPE {
    BORDER
    let p1 = vec3<i32>(p);
    let p2 = min(p1 + vec3<i32>(1), dims - vec3<i32>(1));
    let t = p - vec3<f32>(p1);

    let s111 = BUFFER[vidx(vec3<i32>(p1.x, p1.y, p1.z), dims)];
    let s211 = BUFFER[vidx(vec3<i32>(p2.x, p1.y, p1.z), dims)];
    let s121 = BUFFER[vidx(vec3<i32>(p1.x, p2.y, p1.z), dims)];
    let s221 = BUFFER[vidx(vec3<i32>(p2.x, p2.y, p1.z), dims)];
    let s112 = BUFFER[vidx(vec3<i32>(p1.x, p1.y, p2.z), dims)];
    let s212 = BUFFER[vidx(vec3<i32>(p2.x, p1.y, p2.z), dims)];
    let s122 = BUFFER[vidx(vec3<i32>(p1.x, p2.y, p2.z), dims)];
    let s222 = BUFFER[vidx(vec3<i32>(p2.x, p2.y, p2.z), dims)];

    return (1.0 - t.z) * ((1.0 - t.y) * ((1.0 - t.x) * s111 + t.x * s211)
                        + t.y * ((1.0 - t.x) * s121 + t.x * s221))
         + t.z * ((1.0 - t.y) * ((1.0 - t.x) * s112 + t.x * s212)
                + t.y * ((1.0 - t.x) * s122 + t.x * s222));
}
)";
    if (clamp) {
        replace_all(s, "BORDER",
            "let p = clamp(pos, vec3<f32>(0.0), vec3<f32>(dims - vec3<i32>(1)));");
    }
    else {
        replace_all(s, "BORDER",
            "let p = pos;\n"
            "    if (any(p < vec3<f32>(0.0)) || any(vec3<i32>(p) >= dims)) {\n"
            "        return TYPE(0.0);\n"
            "    }");
    }
    replace_all(s, "NAME", name);
    replace_all(s, "BUFFER", buffer);
    replace_all(s, "TYPE", type);
    return s;
}

std::string storage(int binding, const std::string& name, const std::string& type,
                    bool writable = false)
{
    return "@group(0) @binding(" + std::to_string(binding) + ") var<storage, "
        + (writable ? "read_write" : "read") + "> " + name + ": array<" + type + ">;\n";
}

const char* UNIFORM = "@group(0) @binding(0) var<uniform> params: Params;\n";

// Mirrors CostFunctionKernel, with the exception that the costs before and
// after applying delta are computed by the same invocation.
const char* UNARY_PARAMS = R"(
struct Moving {
    dims: vec4<i32>,
    origin: vec4<f32>,
    inv_spacing: vec4<f32>,
    inv_direction: array<vec4<f32>, 3>,
};

struct Params {
    grid: Grid,
    df: Df,
    moving: Moving,
    delta: vec4<f32>,
    weight: f32,
    update_rule: i32,
    has_fixed_mask: i32,
    has_moving_mask: i32,
    iparams: vec4<i32>,
    fparams: vec4<f32>,
};
)";

const char* UNARY_MAIN = R"(
fn cost_at(p: vec3<i32>, d: vec3<f32>, fixed_mask_value: f32) -> f32 {
    let pt = df_index2point(p) + d;
    let moving_p = mul3(params.moving.inv_direction, pt - params.moving.origin.xyz)
        * params.moving.inv_spacing.xyz;

    // Check if the moving voxels are masked out
    var moving_mask_value = 1.0;
    if (params.has_moving_mask != 0) {
        moving_mask_value = moving_mask_linear_at_border(moving_p, params.moving.dims.xyz);
        if (moving_mask_value <= FLT_EPSILON) {
            return 0.0;
        }
    }
    return cost_impl(p, moving_p, d) * params.weight * fixed_mask_value * moving_mask_value;
}

MAIN_HEADER
    let p = vec3<i32>(gid);
    let dims = params.grid.dims.xyz;
    if (any(p >= dims)) {
        return;
    }

    let blk = block_of(p);
    let slot = block_slot(blk);
    if (slot == NO_SLOT) {
        return;
    }

    // Check if the fixed voxel is masked out
    var fixed_mask_value = 1.0;
    if (params.has_fixed_mask != 0) {
        fixed_mask_value = fixed_mask_vol[vidx(p, dims)];
        if (fixed_mask_value <= FLT_EPSILON) {
            return;
        }
    }

    let c = vec2<f32>(
        cost_at(p, df_get(p), fixed_mask_value),
        cost_at(p, df_get_delta(p, params.delta.xyz, params.update_rule), fixed_mask_value)
    );
    cost_out[unary_index(slot, block_local(p, blk))] += c;
}
)";

const char* IMPL_SSD = R"(
fn cost_impl(p: vec3<i32>, moving_p: vec3<f32>, d: vec3<f32>) -> f32 {
    let diff = fixed_vol[vidx(p, params.grid.dims.xyz)]
        - moving_linear_at_border(moving_p, params.moving.dims.xyz);
    return diff * diff;
}
)";

// iparams.x: radius
//
// WGSL has no 64-bit floats, which the CUDA version uses for the sums. To
// compensate, values are accumulated relative to the values at the center of
// the window. This does not change the result, as the correlation is
// invariant to offsets, but avoids most of the cancellation when subtracting
// the means.
const char* IMPL_NCC = R"(
fn cost_impl(p: vec3<i32>, moving_p: vec3<f32>, d: vec3<f32>) -> f32 {
    let fixed_dims = params.grid.dims.xyz;
    let moving_dims = params.moving.dims.xyz;
    let radius = params.iparams.x;

    let fixed_ref = fixed_vol[vidx(p, fixed_dims)];
    let moving_ref = moving_linear_at_border(moving_p, moving_dims);

    var sff = 0.0;
    var sf = 0.0;
    var smm = 0.0;
    var sfm = 0.0;
    var sm = 0.0;
    var n = 0.0;

    for (var dz = -radius; dz <= radius; dz++) {
    for (var dy = -radius; dy <= radius; dy++) {
    for (var dx = -radius; dx <= radius; dx++) {
        // TODO: Does not account for anisotropic volumes
        let r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > radius * radius) {
            continue;
        }

        let fp = p + vec3<i32>(dx, dy, dz);
        if (any(fp < vec3<i32>(0)) || any(fp >= fixed_dims)) {
            continue;
        }
        let mp = moving_p + vec3<f32>(f32(dx), f32(dy), f32(dz));

        let fixed_v = fixed_vol[vidx(fp, fixed_dims)] - fixed_ref;
        let moving_v = moving_linear_at_border(mp, moving_dims) - moving_ref;

        sff += fixed_v * fixed_v;
        smm += moving_v * moving_v;
        sfm += fixed_v * moving_v;
        sm += moving_v;
        sf += fixed_v;
        n += 1.0;
    }
    }
    }

    if (n == 0.0) {
        return 0.0;
    }

    // Subtract mean
    sff -= sf * sf / n;
    smm -= sm * sm / n;
    sfm -= sf * sm / n;

    let denom = sqrt(sff * smm);

    // Set cost to zero if outside moving volume
    if (all(moving_p >= vec3<f32>(0.0)) && all(moving_p < vec3<f32>(moving_dims))
        && denom > 1e-5) {
        return 0.5 * (1.0 - sfm / denom);
    }
    return 0.0;
}
)";

// iparams.x: number of landmarks, fparams.x: half decay
// The buffer holds the fixed landmarks followed by their displacements.
const char* IMPL_LANDMARKS = R"(
fn cost_impl(p: vec3<i32>, moving_p: vec3<f32>, d: vec3<f32>) -> f32 {
    let epsilon = 1e-6;
    let count = params.iparams.x;
    let half_decay = params.fparams.x;
    let world_p = df_index2point(p);

    var c = 0.0;
    for (var i = 0; i < count; i++) {
        let lm = landmarks[i].xyz;
        let dp = landmarks[count + i].xyz;

        let inv_den = 1.0 / (pow(norm2(lm - world_p), half_decay) + epsilon);
        c += norm2(d - dp) * inv_den;
    }
    return c;
}
)";

const char* IMPL_SOFT_CONSTRAINTS = R"(
fn cost_impl(p: vec3<i32>, moving_p: vec3<f32>, d: vec3<f32>) -> f32 {
    let i = vidx(p, params.grid.dims.xyz);
    // Clamp to avoid explosion
    let dist2 = min(norm2(d - constraint_values[i].xyz), 100000.0);
    return constraint_mask[i] * dist2;
}
)";

const char* BINARY_PARAMS = R"(
struct Params {
    grid: Grid,
    df: Df,
    delta: vec4<f32>,
    inv_spacing2_exp: vec4<f32>,
    weight: f32,
    scale: f32,
    half_exponent: f32,
    update_rule: i32,
    has_weight_map: i32,
};
)";

const char* BINARY_COMMON = R"(
fn energy_term(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let v = params.scale * norm2(a - b);
    if (v <= 0.0) {
        return 0.0;
    }
    return pow(v, params.half_exponent);
}

// {E00, E01, E10, E11}
fn energy(d0: vec3<f32>, d1: vec3<f32>, dn0: vec3<f32>, dn1: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(
        energy_term(d0, dn0),
        energy_term(d0, dn1),
        energy_term(d1, dn0),
        energy_term(d1, dn1)
    );
}
)";

// Computes the cost between a voxel and its neighbour in the positive
// direction of the axis. The cost is stored in the slot of the block of the
// voxel and, if the neighbour is in another block, in the padding of the slot
// of that block. This replaces do_step_border of the CUDA version, which
// computes the same values from the neighbouring block.
const char* BINARY_AXIS = R"(
fn do_step_AXIS(p: vec3<i32>, blk: vec3<i32>, slot: u32, d0: vec3<f32>, d1: vec3<f32>) {
    let dims = params.grid.dims.xyz;
    let axis = STEP;
    let q = p + axis;
    if (q.AXIS >= dims.AXIS) {
        return;
    }

    let q_blk = block_of(q);
    var q_slot = NO_SLOT;
    if (q_blk.AXIS != blk.AXIS) {
        q_slot = block_slot(q_blk);
    }
    if (slot == NO_SLOT && q_slot == NO_SLOT) {
        return;
    }

    var weight = params.weight;
    if (params.has_weight_map != 0) {
        weight = 0.5 * (weight_map[vidx(p, dims)] + weight_map[vidx(q, dims)]);
    }

    let dn0 = df_get(q);
    let dn1 = df_get_delta(q, params.delta.xyz, params.update_rule);
    let e = weight * params.inv_spacing2_exp.AXIS * energy(d0, d1, dn0, dn1);

    let l = block_local(p, blk);
    if (slot != NO_SLOT) {
        cost_AXIS[binary_index(slot, l, axis)] = e;
    }
    if (q_slot != NO_SLOT) {
        var q_l = l;
        q_l.AXIS = -1;
        cost_AXIS[binary_index(q_slot, q_l, axis)] = e;
    }
}
)";

const char* BINARY_MAIN = R"(
MAIN_HEADER
    let p = vec3<i32>(gid);
    if (any(p >= params.grid.dims.xyz)) {
        return;
    }

    let blk = block_of(p);
    let slot = block_slot(blk);

    // Voxels in blocks without a slot are only of interest if they border to
    // another block.
    if (slot == NO_SLOT
        && all(block_local(p, blk) < params.grid.block_dims.xyz - vec3<i32>(1))) {
        return;
    }

    let d0 = df_get(p);
    let d1 = df_get_delta(p, params.delta.xyz, params.update_rule);

    do_step_x(p, blk, slot, d0, d1);
    do_step_y(p, blk, slot, d0, d1);
    do_step_z(p, blk, slot, d0, d1);
}
)";

// Invocations are (slot, z-layer within the slot). The final summation is
// done on the host with double precision.
const char* REDUCE_ENERGY = R"(
struct Params {
    grid: Grid,
    num_slots: i32,
};
UNIFORM
BUFFERS
GRID
MAIN_HEADER
    let b = params.grid.block_dims.xyz;
    if (i32(gid.x) >= params.num_slots || i32(gid.y) >= b.z || gid.z != 0u) {
        return;
    }
    let slot = gid.x;

    var sum = 0.0;
    for (var y = 0; y < b.y; y++) {
    for (var x = 0; x < b.x; x++) {
        let l = vec3<i32>(x, y, i32(gid.y));
        sum += unary_cost[unary_index(slot, l)].x
            + cost_x[binary_index(slot, l, vec3<i32>(1, 0, 0))].x
            + cost_y[binary_index(slot, l, vec3<i32>(0, 1, 0))].x
            + cost_z[binary_index(slot, l, vec3<i32>(0, 0, 1))].x;
    }
    }
    sums[slot * u32(b.z) + gid.y] = sum;
}
)";

const char* APPLY_DELTA = R"(
struct Params {
    df: Df,
    delta: vec4<f32>,
};
UNIFORM
BUFFERS
COMMON
LINEAR_AT
DISPLACEMENT_FIELD
// Labels are bytes, packed four to an element
fn label_at(i: i32) -> u32 {
    return (labels[u32(i) >> 2u] >> ((u32(i) & 3u) * 8u)) & 0xffu;
}

MAIN_HEADER
    let p = vec3<i32>(gid);
    let dims = params.df.dims.xyz;
    if (any(p >= dims)) {
        return;
    }
    let i = vidx(p, dims);
    if (label_at(i) != 0u) {
        df_out[i] = vec4<f32>(df_update(p, params.delta.xyz, UPDATE_RULE), 0.0);
    }
}
)";

const char* COMPUTE_DISPLACEMENT_FIELD = R"(
struct Params {
    df: Df,
};
UNIFORM
BUFFERS
COMMON
LINEAR_AT
DISPLACEMENT_FIELD
MAIN_HEADER
    let p = vec3<i32>(gid);
    let dims = params.df.dims.xyz;
    if (any(p >= dims)) {
        return;
    }
    df_out[vidx(p, dims)] = vec4<f32>(df_get(p), 0.0);
}
)";

const char* FILTER_PARAMS = R"(
struct Params {
    src_dims: vec4<i32>,
    dst_dims: vec4<i32>,
    fparams: vec4<f32>,
    iparams: vec4<i32>,
};
)";

// Shared start of the filters, p is the destination voxel
const char* FILTER_MAIN = R"(
MAIN_HEADER
    let p = vec3<i32>(gid);
    let src_dims = params.src_dims.xyz;
    let dst_dims = params.dst_dims.xyz;
    if (any(p >= dst_dims)) {
        return;
    }
)";

const char* GAUSSIAN_FILTER = R"(
    let factor = params.fparams.x;
    let spacing = params.fparams.y;
    let filter_size = params.iparams.x;

    var axis = vec3<i32>(1, 0, 0);
    if (params.iparams.y == 1) {
        axis = vec3<i32>(0, 1, 0);
    }
    else if (params.iparams.y == 2) {
        axis = vec3<i32>(0, 0, 1);
    }

    var value = TYPE(0.0);
    var norm = 0.0;
    for (var t = -filter_size; t < filter_size + 1; t++) {
        let c = exp(factor * f32(t * t) * spacing * spacing);
        let s = clamp(p + t * axis, vec3<i32>(0), src_dims - vec3<i32>(1));
        value += c * src[vidx(s, src_dims)];
        norm += c;
    }
    dst[vidx(p, dst_dims)] = value / norm;
}
)";

const char* DOWNSAMPLE_MASK = R"(
    var m = 0.0;
    for (var i = 0; i < 8; i++) {
        let q = 2 * p + vec3<i32>(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        if (any(q >= src_dims)) {
            continue;
        }
        m = max(m, mask[vidx(q, src_dims)]);
    }
    dst[vidx(p, dst_dims)] = m;
}
)";

const char* DOWNSAMPLE_VALUES = R"(
    var count = 0.0;
    var value = vec4<f32>(0.0);
    for (var i = 0; i < 8; i++) {
        let q = 2 * p + vec3<i32>(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        if (any(q >= src_dims)) {
            continue;
        }
        if (mask[vidx(q, src_dims)] > 0.0) {
            count += 1.0;
            value += values[vidx(q, src_dims)];
        }
    }
    if (count > 0.0) {
        value = value / count;
    }
    dst[vidx(p, dst_dims)] = value;
}
)";

std::string finalize(std::string s)
{
    replace_all(s, "MAIN_HEADER", MAIN_HEADER);
    replace_all(s, "UNIFORM", UNIFORM);
    return s;
}

std::string filter_shader(const std::string& buffers, const std::string& extra,
                          const std::string& body)
{
    return finalize(std::string(FILTER_PARAMS) + UNIFORM + buffers + COMMON + extra
        + FILTER_MAIN + body);
}

} // namespace

namespace webgpu {

std::string unary_shader(CostFunction fn)
{
    std::string buffers = storage(1, "slots", "u32")
        + storage(2, "df_vol", "vec4<f32>")
        + storage(5, "fixed_mask_vol", "f32")
        + storage(6, "moving_mask_vol", "f32")
        + storage(7, "cost_out", "vec2<f32>", true);

    std::string impl;
    if (fn == CostFunction_SSD || fn == CostFunction_NCC) {
        buffers += storage(3, "fixed_vol", "f32") + storage(4, "moving_vol", "f32");
        impl = linear_at("moving_linear_at_border", "moving_vol", "f32", false)
            + (fn == CostFunction_SSD ? IMPL_SSD : IMPL_NCC);
    }
    else if (fn == CostFunction_Landmarks) {
        buffers += storage(3, "landmarks", "vec4<f32>");
        impl = IMPL_LANDMARKS;
    }
    else {
        buffers += storage(3, "constraint_mask", "f32")
            + storage(4, "constraint_values", "vec4<f32>");
        impl = IMPL_SOFT_CONSTRAINTS;
    }

    return finalize(
        std::string(GRID) + GRID_SLOTS + DISPLACEMENT_FIELD + UNARY_PARAMS + UNIFORM + buffers + COMMON
        + linear_at("df_linear_at_clamp", "df_vol", "vec4<f32>", true)
        + linear_at("moving_mask_linear_at_border", "moving_mask_vol", "f32", false)
        + impl + UNARY_MAIN
    );
}

std::string binary_shader()
{
    std::string buffers = storage(1, "slots", "u32")
        + storage(2, "df_vol", "vec4<f32>")
        + storage(3, "weight_map", "f32")
        + storage(4, "cost_x", "vec4<f32>", true)
        + storage(5, "cost_y", "vec4<f32>", true)
        + storage(6, "cost_z", "vec4<f32>", true);

    std::string steps;
    const char* axes[] = {"x", "y", "z"};
    const char* vectors[] = {
        "vec3<i32>(1, 0, 0)", "vec3<i32>(0, 1, 0)", "vec3<i32>(0, 0, 1)"
    };
    for (int i = 0; i < 3; ++i) {
        std::string step = BINARY_AXIS;
        replace_all(step, "AXIS", axes[i]);
        replace_all(step, "STEP", vectors[i]);
        steps += step;
    }

    return finalize(
        std::string(GRID) + GRID_SLOTS + DISPLACEMENT_FIELD + BINARY_PARAMS + UNIFORM + buffers + COMMON
        + linear_at("df_linear_at_clamp", "df_vol", "vec4<f32>", true)
        + BINARY_COMMON + steps + BINARY_MAIN
    );
}

std::string reduce_energy_shader()
{
    std::string s = REDUCE_ENERGY;
    replace_all(s, "BUFFERS",
        storage(1, "unary_cost", "vec2<f32>")
        + storage(2, "cost_x", "vec4<f32>")
        + storage(3, "cost_y", "vec4<f32>")
        + storage(4, "cost_z", "vec4<f32>")
        + storage(5, "sums", "f32", true));
    replace_all(s, "GRID", GRID);
    return finalize(s);
}

std::string apply_delta_shader(bool compositive)
{
    std::string s = APPLY_DELTA;
    if (compositive) {
        replace_all(s, "BUFFERS",
            storage(1, "labels", "u32")
            + storage(2, "df_vol", "vec4<f32>")
            + storage(3, "df_out", "vec4<f32>", true));
        replace_all(s, "LINEAR_AT",
            linear_at("df_linear_at_clamp", "df_vol", "vec4<f32>", true));
        replace_all(s, "UPDATE_RULE", "UPDATE_RULE_COMPOSITIVE");
    }
    else {
        // All updates are independent, so the field can be updated in place
        replace_all(s, "BUFFERS",
            storage(1, "labels", "u32")
            + storage(2, "df_vol", "vec4<f32>", true));
        replace_all(s, "LINEAR_AT",
            linear_at("df_linear_at_clamp", "df_vol", "vec4<f32>", true));
        replace_all(s, "df_out[i]", "df_vol[i]");
        replace_all(s, "UPDATE_RULE)", "UPDATE_RULE_ADDITIVE)");
    }
    replace_all(s, "DISPLACEMENT_FIELD", DISPLACEMENT_FIELD);
    replace_all(s, "COMMON", COMMON);
    return finalize(s);
}

std::string compute_displacement_field_shader()
{
    std::string s = COMPUTE_DISPLACEMENT_FIELD;
    replace_all(s, "BUFFERS",
        storage(1, "df_vol", "vec4<f32>") + storage(2, "df_out", "vec4<f32>", true));
    replace_all(s, "LINEAR_AT",
        linear_at("df_linear_at_clamp", "df_vol", "vec4<f32>", true));
    replace_all(s, "DISPLACEMENT_FIELD", DISPLACEMENT_FIELD);
    replace_all(s, "COMMON", COMMON);
    return finalize(s);
}

std::string gaussian_filter_shader(const std::string& type)
{
    std::string body = GAUSSIAN_FILTER;
    replace_all(body, "TYPE", type);
    return filter_shader(storage(1, "src", type) + storage(2, "dst", type, true), "", body);
}

std::string shrink_by_2_shader(const std::string& type)
{
    return filter_shader(
        storage(1, "src", type) + storage(2, "dst", type, true), "",
        "    dst[vidx(p, dst_dims)] = src[vidx(2 * p, src_dims)];\n}\n"
    );
}

std::string upsample_vectorfield_shader()
{
    return filter_shader(
        storage(1, "src", "vec4<f32>") + storage(2, "dst", "vec4<f32>", true),
        linear_at("src_linear_at_clamp", "src", "vec4<f32>", true),
        "    dst[vidx(p, dst_dims)] = src_linear_at_clamp(\n"
        "        vec3<f32>(p) * params.fparams.xyz, src_dims);\n}\n"
    );
}

std::string downsample_mask_shader()
{
    return filter_shader(
        storage(1, "mask", "f32") + storage(2, "dst", "f32", true), "", DOWNSAMPLE_MASK);
}

std::string downsample_values_shader()
{
    return filter_shader(
        storage(1, "mask", "f32") + storage(2, "values", "vec4<f32>")
        + storage(3, "dst", "vec4<f32>", true), "", DOWNSAMPLE_VALUES);
}

std::string constrain_displacement_field_shader()
{
    return filter_shader(
        storage(1, "mask", "f32") + storage(2, "values", "vec4<f32>")
        + storage(3, "df_vol", "vec4<f32>", true), "",
        "    let i = vidx(p, dst_dims);\n"
        "    if (mask[i] > 0.5) {\n"
        "        df_vol[i] = values[i];\n"
        "    }\n}\n"
    );
}

} // namespace webgpu

#endif // DF_USE_WEBGPU
