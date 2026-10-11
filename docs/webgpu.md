# WebGPU support

deform can run GPU supported registration through WebGPU, as a cross-platform
alternative to CUDA. WebGPU runs on GPUs from any vendor, through Vulkan
(Linux, Windows), Metal (macOS) or DirectX 12 (Windows).

This document describes how to build and use it, how the implementation is
structured, how it differs from the CUDA implementation, and what is left to
do.

## Status

The registration path is fully implemented, with the same features as the CUDA
engine:

* Cost functions: squared distance (SSD), cross correlation (NCC), landmarks
  and soft constraints
* Fixed and moving masks
* Regularization, with optional weight map
* Additive and compositive update rules
* Affine transform, initial displacement field and voxel constraints
* GCO and ICM solvers

It has been tested on Linux with an AMD integrated GPU (Mesa RADV, Vulkan
backend). It has **not** been tested on macOS or Windows, on other GPU
vendors, or compared against the CUDA engine.

## Building

WebGPU support is disabled by default and enabled with the CMake option
`DF_USE_WEBGPU`. This also enables `STK_USE_WEBGPU` in stk.

```
cmake -S . -B build -DDF_USE_WEBGPU=ON
```

For the Python package:

```
sh build_python.sh -DDF_USE_WEBGPU=ON
```

The implementation is built on
[wgpu-native](https://github.com/gfx-rs/wgpu-native). A prebuilt release for
the platform is downloaded by CMake and linked statically, so nothing has to
be installed and no extra library has to be distributed. Related options:

| Option | Description |
|---|---|
| `WGPU_NATIVE_VERSION` | Release to download (default `v29.0.1.1`) |
| `WGPU_NATIVE_DIR` | Path to an already extracted release, skips the download |

The wheels built with cibuildwheel do not enable WebGPU.

## Usage

GPU supported registration is requested the same way as with CUDA:

```python
df = pydeform.register(fixed, moving, settings=settings, use_gpu=True)
```

or `--gpu` for the executable. `pydeform.has_gpu()` returns true if the build
has either CUDA or WebGPU support. If a build has both, CUDA is used.

The device is selected automatically, preferring a high performance GPU, and
is logged when first used. An error is raised if no device is available.

## Implementation

### stk

| File | Content |
|---|---|
| `src/stk/webgpu/webgpu.h` | Device, `Buffer` and `Kernel` |
| `src/stk/image/webgpu_volume.h` | `stk::WebGpuVolume` |
| `cmake/WgpuNative.cmake` | Download of wgpu-native |
| `test/test_webgpu.cpp` | Tests |

`stk::webgpu` is a thin layer for running compute kernels. The WebGPU headers
are not exposed, so users of stk only see these three concepts:

* **Device**: created on first use and kept for the remainder of the process.
  It requests the limits of the adapter, as the default limits of WebGPU are
  too low (e.g. buffers of 256 MB).
* **`Buffer`**: a reference counted storage buffer. Supports upload, download,
  mapping for reading, clearing and copying.
* **`Kernel`**: a compute shader written in WGSL, with a uniform block at
  binding 0 followed by a number of storage buffers. `run` takes the
  parameters, the buffers, and the extent to run the kernel over.

`stk::WebGpuVolume` is the counterpart to `stk::GpuVolume`. Voxels are stored
contiguously in a buffer, and supported types are float, float2, float4 and
uint8.

None of this is thread-safe; all calls have to be made from the same thread.

### deform

| File | Content |
|---|---|
| `registration/webgpu/shaders.cpp` | WGSL sources for all kernels |
| `registration/webgpu/shaders.h` | Uniform blocks for the kernels |
| `registration/webgpu/block_grid.h` | Layout of the cost buffers |
| `registration/webgpu/cost_functions.cpp` | Unary and binary cost functions |
| `registration/webgpu/displacement_field.cpp` | `WebGpuDisplacementField` |
| `registration/webgpu/voxel_constraints.cpp` | Voxel constraints |
| `filters/webgpu/filters.cpp` | Gaussian filter and resampling |
| `registration/webgpu_graph_cut_optimizer.inl` | Optimizer |
| `registration/webgpu_registration_engine.cpp` | Engine |
| `test/test_webgpu.cpp` | Tests |

The structure mirrors the CUDA implementation (`registration/gpu`,
`filters/gpu`, `GpuRegistrationEngine`, `HybridGraphCutOptimizer`), and the
kernels are direct ports of their CUDA counterparts unless noted below.

#### Shaders

WGSL has no templates and no way of passing a buffer to a function. The CUDA
code relies on templates for selecting cost function, voxel type and update
rule, so the shaders are instead assembled from pieces of source in
`shaders.cpp`:

* Shared pieces for the block grid and the displacement field.
* A generated trilinear interpolation function for each buffer that needs one.
* One implementation of `cost_impl` for each cost function.

The update rule and the presence of masks are parameters in the uniform block
and not separate shaders.

The uniform blocks in `shaders.h` have to match the structs in the shaders.
To keep this simple they only consist of 16 byte vectors and groups of four
scalars.

#### Optimizer

The optimizer is hybrid, as with CUDA: costs are computed on the GPU and the
graph cuts are solved on the CPU. The iteration scheme (shifting, red-black
ordering, block change flags, acceptance of solutions) is the same.

What differs is how costs get from the GPU to the CPU. CUDA computes the costs
block by block into full volumes, on several streams, and downloads the region
of each block. Per-block round trips are not a good fit for WebGPU, so here:

1. All blocks to update for a step are assigned a *slot*.
2. The cost kernels run over the whole volume once, skipping voxels in blocks
   without a slot, and store the costs of each block contiguously in its slot.
3. The buffers are mapped and read directly by the solvers, which run in
   parallel over the blocks with OpenMP.

This keeps the transfers to what is needed, in a few large reads. The slots
for the binary costs are padded by one voxel in the negative direction of
their axis, holding the cost between the block and its neighbour. This
replaces `do_step_border` in the CUDA version.

The cost buffers have a budget of 128 MB each. If the blocks of a step do not
fit they are processed in several batches.

The total energy is computed with the same kernels, followed by a reduction on
the GPU to one partial sum for each layer of each block. The final summation
is done on the CPU in double precision.

## Differences from the CUDA implementation

* **Cross correlation**: WGSL has no 64-bit floats, which the CUDA version
  uses for the sums. Values are instead accumulated relative to the values at
  the center of the window. The result is the same, as the correlation is
  invariant to offsets, but most of the cancellation is avoided.
* **Constraint mask**: kept as a float volume on the device, as WGSL has no
  8-bit types.
* **Unary costs**: the costs before and after applying delta are computed in
  one pass, instead of two kernel launches.
* **Interpolation**: all interpolation is done manually in the shaders. No
  hardware samplers or textures are used.
* **No overlap of GPU and CPU work**: the optimizer waits for the costs before
  solving. CUDA overlaps the two using streams and a worker pool.

Not ported, as the registration does not use them:

* `gpu::transform_volume`
* The GPU normalize filter in stk

## Limitations

* **Volume size**: each volume has to fit in a single buffer. The limit
  depends on the GPU and is commonly 2 GB, corresponding to a displacement
  field of about 500^3 voxels. An error is raised if the limit is exceeded.
* **Precision**: all computations on the GPU are in single precision.
* **Platforms**: the link settings for macOS and Windows in
  `WgpuNative.cmake` have never been built.

## Testing

The tests are part of the regular test executables and are skipped if no
device is available:

```
cmake -S . -B build -DDF_USE_WEBGPU=ON -DDF_BUILD_TESTS=ON
cmake --build build
cd test && ../build/bin/deform_test "[webgpu]"
```

They cover:

* Each cost function, the regularizer and the filters, compared voxel by voxel
  to the CPU implementation. The cost functions are computed through a shifted
  grid of blocks that does not divide the volume evenly, to exercise the slot
  layout and the padding.
* Full registrations, compared to the CPU engine on the quality of the result.

For a given displacement field the WebGPU and CPU engines report the same
energy. The results of a registration are however not the same. The WebGPU
optimizer applies the moves one direction at a time, as the CUDA optimizer
does, while the CPU optimizer does not. In the tests done so far the WebGPU
engine runs for more iterations and reaches a lower energy.

Also note that the soft constraints are clamped at 100000 here and in CUDA,
but at 1000 in the CPU implementation.

## Performance

Measured on the integrated GPU mentioned above, against the CPU engine on 12
threads, for a 128^3 volume with default settings and 4 pyramid levels:

| | CPU | WebGPU |
|---|---|---|
| SSD | 19.5 s | 20.5 s |
| NCC | 9.5 s | 5.7 s |

The engines do not run the same number of iterations, so this is not a like
for like comparison. No measurements have been made on a discrete GPU.

## Future work

* Test on macOS, Windows and discrete GPUs, and compare results and speed with
  CUDA on the same hardware.
* Overlap the cost computation of one batch with the minimization of the
  previous one.
* Avoid running the cost kernels over the whole volume when only a few blocks
  are active.
* Enable WebGPU in the wheels.
* Lift the size limit, by splitting volumes over several buffers or by using
  3D textures.
