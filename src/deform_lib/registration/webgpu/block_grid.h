#pragma once

#include <deform_lib/config.h>

#ifdef DF_USE_WEBGPU

#include "shaders.h"

#include <stk/math/int3.h>
#include <stk/webgpu/webgpu.h>

#include <cstddef>
#include <cstdint>

namespace webgpu {

// Regular grid of blocks covering a volume, optionally shifted.
//
// The cost kernels only compute costs for blocks that have been assigned a
// slot, and the costs for a block are stored contiguously in that slot. This
// keeps the amount of data that needs to be transferred to the host to a
// minimum.
//
// The slots for the binary costs are padded by one voxel in the negative
// direction of their axis, holding the cost between the block and its
// neighbour.
struct BlockGrid
{
    static const uint32_t NoSlot = 0xffffffffu;

    int3 dims;
    int3 block_dims;
    int3 block_offset;
    int3 block_count;

    int num_blocks() const
    {
        return block_count.x * block_count.y * block_count.z;
    }
    int block_index(const int3& block) const
    {
        return block.x + block_count.x * (block.y + block_count.y * block.z);
    }

    // First voxel of the block, may be outside the volume for shifted grids
    int3 block_begin(const int3& block) const
    {
        return int3{
            block.x * block_dims.x - block_offset.x,
            block.y * block_dims.y - block_offset.y,
            block.z * block_dims.z - block_offset.z
        };
    }

    // Number of elements in a slot
    size_t unary_slot_size() const
    {
        return size_t(block_dims.x) * block_dims.y * block_dims.z;
    }
    size_t binary_slot_size(int axis) const
    {
        return size_t(block_dims.x + (axis == 0)) * (block_dims.y + (axis == 1))
            * (block_dims.z + (axis == 2));
    }

    // Index into the cost buffers
    // l : Coordinates of the voxel within the block
    size_t unary_index(uint32_t slot, const int3& l) const
    {
        return slot * unary_slot_size()
            + l.x + size_t(block_dims.x) * (l.y + size_t(block_dims.y) * l.z);
    }
    // l : Coordinates of the voxel within the block, -1 is allowed for `axis`
    size_t binary_index(int axis, uint32_t slot, const int3& l) const
    {
        const size_t sx = block_dims.x + (axis == 0);
        const size_t sy = block_dims.y + (axis == 1);
        const size_t lx = l.x + (axis == 0);
        const size_t ly = l.y + (axis == 1);
        const size_t lz = l.z + (axis == 2);
        return slot * binary_slot_size(axis) + lx + sx * (ly + sy * lz);
    }

    GridParams params() const
    {
        return GridParams{
            {dims.x, dims.y, dims.z, 0},
            {block_dims.x, block_dims.y, block_dims.z, 0},
            {block_offset.x, block_offset.y, block_offset.z, 0},
            {block_count.x, block_count.y, block_count.z, 0}
        };
    }
};

// Destination for the cost functions
struct CostBuffers
{
    BlockGrid grid;

    // Slot for each block in the grid (uint32_t), or BlockGrid::NoSlot
    stk::webgpu::Buffer slots;
    // float2 with E0 and E1
    stk::webgpu::Buffer unary;
    // float4 with {E00, E01, E10, E11}, for x+, y+ and z+
    stk::webgpu::Buffer binary[3];
};

} // namespace webgpu

#endif // DF_USE_WEBGPU
