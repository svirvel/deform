#pragma once

#include <deform_lib/config.h>

#ifdef DF_USE_WEBGPU

#include <stk/image/volume.h>
#include <stk/webgpu/webgpu.h>

#include "webgpu/block_grid.h"
#include "webgpu/displacement_field.h"
#include "block_change_flags.h"
#include "settings.h"

#include <vector>

class WebGpuBinaryFunction;
class WebGpuUnaryFunction;

// Hybrid (CPU-GPU) graph-cut optimizer, WebGPU version of HybridGraphCutOptimizer.
//  Uses GPU to compute weights for graph and minimizes graph on CPU.
//
// Where the CUDA version dispatches the cost computation block by block on
// multiple streams, this version computes the costs for a batch of blocks at
// a time, into buffers holding only the costs of those blocks.
template<typename TSolver>
class WebGpuGraphCutOptimizer
{
public:
    WebGpuGraphCutOptimizer(
        const std::vector<int3>& neighborhood,
        const Settings::Level& settings,
        Settings::UpdateRule update_rule,
        WebGpuUnaryFunction& unary_fn,
        WebGpuBinaryFunction& binary_fn,
        WebGpuDisplacementField& df
    );
    ~WebGpuGraphCutOptimizer();

    void execute();

private:
    struct Block
    {
        int3 idx;
        int3 begin;
        int3 end;
        bool shift;
    };

    // Allocates the buffers for the costs
    // max_block_count : Largest number of blocks in any grid
    void allocate_cost_buffers(const int3& block_dims, size_t max_block_count);

    // Computes the costs for the given blocks, where each block is assigned
    // the slot corresponding to its position in the range.
    void compute_costs(
        const webgpu::BlockGrid& grid,
        const Block* blocks,
        size_t num_blocks,
        const float3& delta,
        Settings::UpdateRule update_rule
    );

    // Dispatches all queued blocks
    // Returns the number of changed blocks
    size_t dispatch_blocks(const webgpu::BlockGrid& grid);

    // Performs graph cut on block
    // Returns true if block was changed
    bool minimize_block(
        const webgpu::BlockGrid& grid,
        const Block& block,
        uint32_t slot,
        const float2* unary_cost,
        const float4* binary_cost_x,
        const float4* binary_cost_y,
        const float4* binary_cost_z
    );

    // Applies delta based on labels
    void apply_displacement_delta();

    // Computes the total energy of the current displacement field
    double calculate_energy();

    std::vector<int3> _neighborhood;

    const Settings::Level& _settings;
    Settings::UpdateRule _update_rule;

    // Buffers for the costs and the number of slots they hold
    webgpu::CostBuffers _cost;
    size_t _slot_capacity;

    // Partial energy sums
    stk::webgpu::Buffer _energy_sums;

    // Labels from the minimization
    stk::VolumeUChar _labels;
    stk::webgpu::Buffer _gpu_labels;

    WebGpuUnaryFunction& _unary_fn;
    WebGpuBinaryFunction& _binary_fn;

    WebGpuDisplacementField _df;
    WebGpuDisplacementField _df_tmp;

    BlockChangeFlags _block_change_flags;

    std::vector<Block> _block_queue;

    float3 _current_delta;
};

#include "webgpu_graph_cut_optimizer.inl"

#endif // DF_USE_WEBGPU
