#include "webgpu/cost_functions.h"
#include "webgpu/shaders.h"

#include "deform_lib/profiler/profiler.h"

#include <stk/common/log.h>

#include <algorithm>
#include <iomanip>

namespace webgpu_optimizer {
    // Budget for each of the cost buffers, to bound the memory use and the
    // size of the transfers.
    const uint64_t cost_buffer_budget = 128 * 1024 * 1024;
}

template<typename TSolver>
WebGpuGraphCutOptimizer<TSolver>::WebGpuGraphCutOptimizer(
    const std::vector<int3>& neighborhood,
    const Settings::Level& settings,
    Settings::UpdateRule update_rule,
    WebGpuUnaryFunction& unary_fn,
    WebGpuBinaryFunction& binary_fn,
    WebGpuDisplacementField& df) :
    _neighborhood(neighborhood),
    _settings(settings),
    _update_rule(update_rule),
    _slot_capacity(0),
    _unary_fn(unary_fn),
    _binary_fn(binary_fn),
    _df(df),
    _current_delta{0, 0, 0}
{
    // If we're using compositive accumulation we need to have a separate buffer
    //  for the previous iteration since each update may depend on neighbouring
    //  updates. As compared to additive were we only have d(x) = d(x) + delta.

    if (_update_rule == Settings::UpdateRule_Compositive) {
        _df_tmp = _df.clone();
    }

    dim3 dims = _df.size();
    _labels = stk::VolumeUChar(dims, uint8_t{0});
    _gpu_labels = stk::webgpu::Buffer(uint64_t(dims.x) * dims.y * dims.z);
}
template<typename TSolver>
WebGpuGraphCutOptimizer<TSolver>::~WebGpuGraphCutOptimizer()
{
}
template<typename TSolver>
void WebGpuGraphCutOptimizer<TSolver>::execute()
{
    PROFILER_SCOPE("execute", 0xFF532439);

    dim3 dims = _df.size();

    // Setting the block size to (0, 0, 0) will disable blocking and run the whole volume
    int3 block_dims = _settings.block_size;
    if (block_dims.x == 0) block_dims.x = dims.x;
    if (block_dims.y == 0) block_dims.y = dims.y;
    if (block_dims.z == 0) block_dims.z = dims.z;

    int3 block_count {
        int(dims.x + block_dims.x - 1) / block_dims.x,
        int(dims.y + block_dims.y - 1) / block_dims.y,
        int(dims.z + block_dims.z - 1) / block_dims.z
    };

    DLOG(Info) << "Volume size: " << dims;
    DLOG(Info) << "Block count: " << block_count;
    DLOG(Info) << "Block size: " << block_dims;

    // Shifting may add a block in each direction
    allocate_cost_buffers(
        block_dims,
        size_t(block_count.x + 1) * (block_count.y + 1) * (block_count.z + 1)
    );

    _block_change_flags = BlockChangeFlags(block_count);

    LOG(Info) << "Initial Energy: " << calculate_energy();

    int num_iterations = 0;

    bool done = false;
    while (!done) {
        PROFILER_SCOPE("iteration", 0xFF39842A);

        // A max_iteration_count of -1 means we run until we converge
        if (_settings.max_iteration_count != -1 && num_iterations >= _settings.max_iteration_count)
            break;

        done = true;
        size_t num_blocks_changed = 0;

        for (int use_shift = 0; use_shift < 2; ++use_shift) {
        PROFILER_SCOPE("shift", 0xFF766952);
        if (use_shift == 1 && (block_count.x * block_count.y * block_count.z) <= 1)
            continue;

        // We only do shifting in the directions that requires it
        int3 block_offset{0, 0, 0};
        int3 real_block_count = block_count;
        if (use_shift == 1) {
            block_offset.x = block_count.x == 1 ? 0 : (block_dims.x / 2);
            block_offset.y = block_count.y == 1 ? 0 : (block_dims.y / 2);
            block_offset.z = block_count.z == 1 ? 0 : (block_dims.z / 2);

            // Only add an additional shift block if necessary, some configurations may not
            //  need the additional block. E.g. a volume of size 5 with a block size of 4 will
            //  cover whole volume with 2 blocks both with and without shifting.

            if (block_count.x > 1 && (block_count.x * block_dims.x - block_offset.x) < (int)dims.x)
                real_block_count.x += 1;
            if (block_count.y > 1 && (block_count.y * block_dims.y - block_offset.y) < (int)dims.y)
                real_block_count.y += 1;
            if (block_count.z > 1 && (block_count.z * block_dims.z - block_offset.z) < (int)dims.z)
                real_block_count.z += 1;
        }

        webgpu::BlockGrid grid {
            int3{int(dims.x), int(dims.y), int(dims.z)},
            block_dims,
            block_offset,
            real_block_count
        };

        // Only do red-black when having more than 1 block
        int num_blocks = real_block_count.x * real_block_count.y * real_block_count.z;
        for (int black_or_red = 0; black_or_red < (num_blocks > 1 ? 2 : 1); black_or_red++) {
        PROFILER_SCOPE("red_black", 0xFF339955);

        for (int3 n : _neighborhood) {
        PROFILER_SCOPE("step", 0xFFAA6FE2);

        // delta in [mm]
        _current_delta = {
            _settings.step_size.x * n.x,
            _settings.step_size.y * n.y,
            _settings.step_size.z * n.z
        };

        // Queue all blocks
        for (int b = 0; b < num_blocks; ++b) {
            int block_x = b % real_block_count.x;
            int block_y = (b / real_block_count.x) % real_block_count.y;
            int block_z = b / (real_block_count.x*real_block_count.y);

            int off = (block_z) % 2;
            off = (block_y + off) % 2;
            off = (block_x + off) % 2;

            if (off != black_or_red) {
                continue;
            }

            int3 block_idx{block_x, block_y, block_z};

            bool need_update = _block_change_flags.is_block_set(block_idx,
                                                                use_shift == 1);
            for (int3 n2 : _neighborhood) {
                int3 neighbor = block_idx + n2;
                if (0 <= neighbor.x && neighbor.x < real_block_count.x &&
                    0 <= neighbor.y && neighbor.y < real_block_count.y &&
                    0 <= neighbor.z && neighbor.z < real_block_count.z) {
                    need_update = need_update || _block_change_flags.is_block_set(
                        neighbor, use_shift == 1);
                }
            }

            if (!need_update) {
                continue;
            }

            int3 offset = grid.block_begin(block_idx);

            Block block;
            block.idx = block_idx;
            block.begin = int3{
                std::max<int>(0, offset.x),
                std::max<int>(0, offset.y),
                std::max<int>(0, offset.z)
            };
            block.end = int3{
                std::min<int>(offset.x + block_dims.x, dims.x),
                std::min<int>(offset.y + block_dims.y, dims.y),
                std::min<int>(offset.z + block_dims.z, dims.z)
            };
            block.shift = use_shift == 1;

            _block_queue.push_back(block);
        }
        num_blocks_changed += dispatch_blocks(grid);
        }
        }
        }

        done = num_blocks_changed == 0;

        LOG(Verbose) << "Iteration " << num_iterations << ", "
                     << "Changed " << num_blocks_changed << " blocks, "
                     << "Energy " << std::fixed << std::setprecision(9)
                     << calculate_energy();

        ++num_iterations;

        PROFILER_FLIP();
    }
    LOG(Info) << "Energy: "
              << calculate_energy()
              << ", " << "Iterations: " << num_iterations;
}
template<typename TSolver>
void WebGpuGraphCutOptimizer<TSolver>::allocate_cost_buffers(
    const int3& block_dims,
    size_t max_block_count)
{
    webgpu::BlockGrid grid {int3{0, 0, 0}, block_dims, int3{0, 0, 0}, int3{0, 0, 0}};

    size_t max_slot_size = std::max(
        grid.binary_slot_size(0),
        std::max(grid.binary_slot_size(1), grid.binary_slot_size(2))
    );

    uint64_t budget = std::min(
        webgpu_optimizer::cost_buffer_budget, stk::webgpu::max_buffer_size());

    // A single block may exceed the budget, e.g. if blocking is disabled
    _slot_capacity = std::max<size_t>(1, budget / (max_slot_size * sizeof(float4)));
    _slot_capacity = std::min(_slot_capacity, max_block_count);

    _cost.slots = stk::webgpu::Buffer(max_block_count * sizeof(uint32_t));
    _cost.unary = stk::webgpu::Buffer(
        _slot_capacity * grid.unary_slot_size() * sizeof(float2));
    for (int i = 0; i < 3; ++i) {
        _cost.binary[i] = stk::webgpu::Buffer(
            _slot_capacity * grid.binary_slot_size(i) * sizeof(float4));
    }
    _energy_sums = stk::webgpu::Buffer(_slot_capacity * block_dims.z * sizeof(float));
}
template<typename TSolver>
void WebGpuGraphCutOptimizer<TSolver>::compute_costs(
    const webgpu::BlockGrid& grid,
    const Block* blocks,
    size_t num_blocks,
    const float3& delta,
    Settings::UpdateRule update_rule)
{
    ASSERT(num_blocks <= _slot_capacity);

    std::vector<uint32_t> slots(grid.num_blocks(), webgpu::BlockGrid::NoSlot);
    for (size_t i = 0; i < num_blocks; ++i) {
        slots[grid.block_index(blocks[i].idx)] = uint32_t(i);
    }

    _cost.grid = grid;
    _cost.slots.upload(slots.data(), slots.size() * sizeof(uint32_t));

    // Costs are accumulated, and not all entries are written
    _cost.unary.clear(num_blocks * grid.unary_slot_size() * sizeof(float2));
    for (int i = 0; i < 3; ++i) {
        _cost.binary[i].clear(num_blocks * grid.binary_slot_size(i) * sizeof(float4));
    }

    _unary_fn(_df, delta, update_rule, _cost);
    _binary_fn(_df, delta, update_rule, _cost);
}
template<typename TSolver>
size_t WebGpuGraphCutOptimizer<TSolver>::dispatch_blocks(const webgpu::BlockGrid& grid)
{
    size_t num_blocks_changed = 0;

    // Reset labels
    _labels.fill(0);

    for (size_t begin = 0; begin < _block_queue.size(); begin += _slot_capacity) {
        const size_t n = std::min(_slot_capacity, _block_queue.size() - begin);
        const Block* blocks = &_block_queue[begin];

        {
            PROFILER_SCOPE("block_cost", 0xFF532439);
            compute_costs(grid, blocks, n, _current_delta, _update_rule);
        }

        const float2* unary_cost = static_cast<const float2*>(
            _cost.unary.map_read(n * grid.unary_slot_size() * sizeof(float2)));
        const float4* binary_cost[3];
        for (int i = 0; i < 3; ++i) {
            binary_cost[i] = static_cast<const float4*>(
                _cost.binary[i].map_read(n * grid.binary_slot_size(i) * sizeof(float4)));
        }

        std::vector<uint8_t> changed(n, 0);
        {
            PROFILER_SCOPE("minimize_block", 0xFF228844);
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < int(n); ++i) {
                changed[i] = minimize_block(
                    grid, blocks[i], uint32_t(i),
                    unary_cost, binary_cost[0], binary_cost[1], binary_cost[2]
                ) ? 1 : 0;
            }
        }

        _cost.unary.unmap();
        for (int i = 0; i < 3; ++i) {
            _cost.binary[i].unmap();
        }

        for (size_t i = 0; i < n; ++i) {
            if (changed[i]) {
                _block_change_flags.set_block(blocks[i].idx, true, blocks[i].shift);
                ++num_blocks_changed;
            }
        }
    }
    _block_queue.clear();

    if (num_blocks_changed > 0) {
        PROFILER_SCOPE("apply", 0xFF532439);
        apply_displacement_delta();
    }

    return num_blocks_changed;
}
template<typename TSolver>
bool WebGpuGraphCutOptimizer<TSolver>::minimize_block(
    const webgpu::BlockGrid& grid,
    const Block& block,
    uint32_t slot,
    const float2* unary_cost,
    const float4* binary_cost_x,
    const float4* binary_cost_y,
    const float4* binary_cost_z)
{
    dim3 full_dims = _labels.size();

    int3 block_dims {
        block.end.x - block.begin.x,
        block.end.y - block.begin.y,
        block.end.z - block.begin.z
    };

    // Offset from coordinates within this (possibly clipped) block, to
    // coordinates within the slot of the block.
    const int3 slot_begin = grid.block_begin(block.idx);
    const int3 slot_offset {
        block.begin.x - slot_begin.x,
        block.begin.y - slot_begin.y,
        block.begin.z - slot_begin.z
    };

    TSolver graph(block_dims);

    double current_energy = 0;
    {
    for (int sub_z = 0; sub_z < block_dims.z; ++sub_z) {
    for (int sub_y = 0; sub_y < block_dims.y; ++sub_y) {
    for (int sub_x = 0; sub_x < block_dims.x; ++sub_x) {
        // Global coordinates
        int gx = block.begin.x + sub_x;
        int gy = block.begin.y + sub_y;
        int gz = block.begin.z + sub_z;

        // Coordinates within slot
        const int3 l {
            slot_offset.x + sub_x,
            slot_offset.y + sub_y,
            slot_offset.z + sub_z
        };

        const float2& unary = unary_cost[grid.unary_index(slot, l)];
        const float4& binary_x = binary_cost_x[grid.binary_index(0, slot, l)];
        const float4& binary_y = binary_cost_y[grid.binary_index(1, slot, l)];
        const float4& binary_z = binary_cost_z[grid.binary_index(2, slot, l)];

        double f0 = unary.x;
        double f1 = unary.y;

        // Block borders (excl image borders) (T-weights with binary term for neighboring voxels)

        if (sub_x == 0 && gx != 0) {
            const float4& c = binary_cost_x[grid.binary_index(0, slot, int3{l.x-1, l.y, l.z})];
            f0 += c.x;
            f1 += c.y;
        }
        else if (sub_x == block_dims.x - 1 && gx < int(full_dims.x) - 1) {
            f0 += binary_x.x;
            f1 += binary_x.z;
        }

        if (sub_y == 0 && gy != 0) {
            const float4& c = binary_cost_y[grid.binary_index(1, slot, int3{l.x, l.y-1, l.z})];
            f0 += c.x;
            f1 += c.y;
        }
        else if (sub_y == block_dims.y - 1 && gy < int(full_dims.y) - 1) {
            f0 += binary_y.x;
            f1 += binary_y.z;
        }

        if (sub_z == 0 && gz != 0) {
            const float4& c = binary_cost_z[grid.binary_index(2, slot, int3{l.x, l.y, l.z-1})];
            f0 += c.x;
            f1 += c.y;
        }
        else if (sub_z == block_dims.z - 1 && gz < int(full_dims.z) - 1) {
            f0 += binary_z.x;
            f1 += binary_z.z;
        }

        graph.add_term1(sub_x, sub_y, sub_z, f0, f1);

        current_energy += f0;

        if (sub_x + 1 < block_dims.x && gx + 1 < int(full_dims.x)) {
            graph.add_term2(
                sub_x, sub_y, sub_z,
                sub_x + 1, sub_y, sub_z,
                binary_x.x, binary_x.y, binary_x.z, binary_x.w);

            current_energy += binary_x.x;
        }
        if (sub_y + 1 < block_dims.y && gy + 1 < int(full_dims.y)) {
            graph.add_term2(
                sub_x, sub_y, sub_z,
                sub_x, sub_y + 1, sub_z,
                binary_y.x, binary_y.y, binary_y.z, binary_y.w);

            current_energy += binary_y.x;
        }
        if (sub_z + 1 < block_dims.z && gz + 1 < int(full_dims.z)) {
            graph.add_term2(
                sub_x, sub_y, sub_z,
                sub_x, sub_y, sub_z + 1,
                binary_z.x, binary_z.y, binary_z.z, binary_z.w);

            current_energy += binary_z.x;
        }
    }
    }
    }
    }

    double current_emin = graph.minimize();

    bool changed_flag = false;

    if (1.0 - current_emin / current_energy > _settings.block_energy_epsilon) // Accept solution
    {
    for (int sub_z = 0; sub_z < block_dims.z; ++sub_z) {
    for (int sub_y = 0; sub_y < block_dims.y; ++sub_y) {
    for (int sub_x = 0; sub_x < block_dims.x; ++sub_x) {
        // Global coordinates
        int gx = block.begin.x + sub_x;
        int gy = block.begin.y + sub_y;
        int gz = block.begin.z + sub_z;

        if (graph.get_var(sub_x, sub_y, sub_z) == 1) {
            _labels(gx,gy,gz) = 1;
            changed_flag = true;
        }
    }
    }
    }
    }
    return changed_flag;
}
template<typename TSolver>
void WebGpuGraphCutOptimizer<TSolver>::apply_displacement_delta()
{
    using stk::webgpu::Access_Read;
    using stk::webgpu::Access_ReadWrite;

    static stk::webgpu::Kernel additive_kernel(
        webgpu::apply_delta_shader(false),
        {Access_Read, Access_ReadWrite},
        sizeof(webgpu::ApplyDeltaParams)
    );
    static stk::webgpu::Kernel compositive_kernel(
        webgpu::apply_delta_shader(true),
        {Access_Read, Access_Read, Access_ReadWrite},
        sizeof(webgpu::ApplyDeltaParams)
    );

    dim3 dims = _df.size();
    _gpu_labels.upload(_labels.ptr(), uint64_t(dims.x) * dims.y * dims.z);

    webgpu::ApplyDeltaParams params = {};
    params.df = _df.params();
    params.delta[0] = _current_delta.x;
    params.delta[1] = _current_delta.y;
    params.delta[2] = _current_delta.z;

    if (_update_rule == Settings::UpdateRule_Compositive) {
        // Each update may depend on neighbouring updates, so we need to read
        //  from a copy of the previous iteration.
        _df_tmp.copy_from(_df);
        compositive_kernel.run(
            &params,
            {_gpu_labels, _df_tmp.volume().buffer(), _df.volume().buffer()},
            dims
        );
    }
    else {
        additive_kernel.run(&params, {_gpu_labels, _df.volume().buffer()}, dims);
    }
}
template<typename TSolver>
double WebGpuGraphCutOptimizer<TSolver>::calculate_energy()
{
    using stk::webgpu::Access_Read;
    using stk::webgpu::Access_ReadWrite;

    static stk::webgpu::Kernel reduce_kernel(
        webgpu::reduce_energy_shader(),
        {Access_Read, Access_Read, Access_Read, Access_Read, Access_ReadWrite},
        sizeof(webgpu::ReduceParams)
    );

    dim3 dims = _df.size();

    int3 block_dims = _settings.block_size;
    if (block_dims.x == 0) block_dims.x = dims.x;
    if (block_dims.y == 0) block_dims.y = dims.y;
    if (block_dims.z == 0) block_dims.z = dims.z;

    webgpu::BlockGrid grid {
        int3{int(dims.x), int(dims.y), int(dims.z)},
        block_dims,
        int3{0, 0, 0},
        int3{
            int(dims.x + block_dims.x - 1) / block_dims.x,
            int(dims.y + block_dims.y - 1) / block_dims.y,
            int(dims.z + block_dims.z - 1) / block_dims.z
        }
    };

    // The energy is the sum of the costs of all blocks
    std::vector<Block> blocks(grid.num_blocks());
    for (int i = 0; i < grid.num_blocks(); ++i) {
        blocks[i].idx = int3{
            i % grid.block_count.x,
            (i / grid.block_count.x) % grid.block_count.y,
            i / (grid.block_count.x * grid.block_count.y)
        };
    }

    double total_energy = 0;
    std::vector<float> sums;

    for (size_t begin = 0; begin < blocks.size(); begin += _slot_capacity) {
        const size_t n = std::min(_slot_capacity, blocks.size() - begin);

        // Update rule doesn't matter in this case since we don't want the energy for a move.
        compute_costs(grid, &blocks[begin], n, float3{0, 0, 0}, Settings::UpdateRule_Additive);

        webgpu::ReduceParams params = {};
        params.grid = grid.params();
        params.num_slots = int32_t(n);

        reduce_kernel.run(
            &params,
            {_cost.unary, _cost.binary[0], _cost.binary[1], _cost.binary[2], _energy_sums},
            dim3{uint32_t(n), uint32_t(block_dims.z), 1}
        );

        sums.resize(n * block_dims.z);
        _energy_sums.download(sums.data(), sums.size() * sizeof(float));

        for (float s : sums) {
            total_energy += s;
        }
    }
    return total_energy;
}
