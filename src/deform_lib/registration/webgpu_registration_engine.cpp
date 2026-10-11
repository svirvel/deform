#include "webgpu_registration_engine.h"

#ifdef DF_USE_WEBGPU

#include "deform_lib/make_unique.h"
#include "deform_lib/profiler/profiler.h"

#ifdef DF_ENABLE_GCO
    #include "deform_lib/solver/gco_solver.h"
#endif
#ifdef DF_ENABLE_GRIDCUT
    #include "deform_lib/solver/gridcut_solver.h"
#endif
#include "deform_lib/solver/icm_solver.h"

#include "webgpu/cost_functions.h"
#include "webgpu/displacement_field.h"
#include "webgpu/voxel_constraints.h"
#include "webgpu_graph_cut_optimizer.h"

#include "../filters/webgpu/filters.h"

#include <stk/common/log.h>
#include <stk/webgpu/webgpu.h>

namespace {
    template<typename T>
    static T str_to_num(const std::string& f, const std::string& k, const std::string& v) {
        try {
            return static_cast<T>(std::stod(v));
        }
        catch (std::invalid_argument&) {
            throw std::invalid_argument(f + ": unrecognised value "
                                        "'" + v + "' for parameter '" + k + "'");
        }
    }

    stk::VolumeFloat4 volume_float3_to_float4(const stk::VolumeFloat3& df)
    {
        dim3 dims = df.size();
        stk::VolumeFloat4 out(dims);
        out.copy_meta_from(df);

        for (int z = 0; z < (int)dims.z; ++z) {
        for (int y = 0; y < (int)dims.y; ++y) {
        for (int x = 0; x < (int)dims.x; ++x) {
            out(x,y,z) = float4{df(x,y,z).x, df(x,y,z).y, df(x,y,z).z, 0.0f};
        }}}

        return out;
    }
    stk::VolumeFloat3 volume_float4_to_float3(const stk::VolumeFloat4& df)
    {
        dim3 dims = df.size();
        stk::VolumeFloat3 out(dims);
        out.copy_meta_from(df);

        for (int z = 0; z < (int)dims.z; ++z) {
        for (int y = 0; y < (int)dims.y; ++y) {
        for (int x = 0; x < (int)dims.x; ++x) {
            out(x,y,z) = float3{df(x,y,z).x, df(x,y,z).y, df(x,y,z).z};
        }}}

        return out;
    }
    // WGSL has no 8-bit types, so masks are kept as float on the device
    stk::VolumeFloat volume_uchar_to_float(const stk::VolumeUChar& mask)
    {
        dim3 dims = mask.size();
        stk::VolumeFloat out(dims);
        out.copy_meta_from(mask);

        for (int z = 0; z < (int)dims.z; ++z) {
        for (int y = 0; y < (int)dims.y; ++y) {
        for (int x = 0; x < (int)dims.x; ++x) {
            out(x,y,z) = float(mask(x,y,z));
        }}}

        return out;
    }

    bool has_level(const std::vector<stk::WebGpuVolume>& pyramid, int level)
    {
        return level < (int) pyramid.size() && pyramid[level].valid();
    }
}

WebGpuRegistrationEngine::Pyramid WebGpuRegistrationEngine::build_pyramid(
    const stk::WebGpuVolume& base,
    DownsampleFn downsample_fn) const
{
    ASSERT(base.valid());
    ASSERT(_settings.num_pyramid_levels > 0);

    Pyramid pyramid(_settings.num_pyramid_levels);
    pyramid[0] = base;
    for (int i = 0; i < _settings.num_pyramid_levels-1; ++i) {
        pyramid[i+1] = downsample_fn(pyramid[i]);
    }
    return pyramid;
}

void WebGpuRegistrationEngine::build_unary_function(int level, WebGpuUnaryFunction& unary_fn)
{
    using FunctionPtr = std::unique_ptr<WebGpuCostFunction>;

    for (int i = 0; i < (int) _fixed_pyramids.size(); ++i) {
        if (!has_level(_fixed_pyramids[i], level) || !has_level(_moving_pyramids[i], level))
            continue; // Skip empty slots

        stk::WebGpuVolume fixed = _fixed_pyramids[i][level];
        stk::WebGpuVolume moving = _moving_pyramids[i][level];

        ASSERT(fixed.voxel_type() == stk::Type_Float);
        ASSERT(moving.voxel_type() == stk::Type_Float);

        for (auto& fn : _settings.image_slots[i].cost_functions) {
            FunctionPtr function;

            if (Settings::ImageSlot::CostFunction_SSD == fn.function) {
                if (!fn.parameters.empty()) {
                    throw std::invalid_argument("[WebGPU] SSDFunction: unrecognised parameter "
                                                "'" + fn.parameters.begin()->first + "' with value '"
                                                + fn.parameters.begin()->second + "'");
                }

                function = make_unique<WebGpuCostFunction_SSD>(fixed, moving);
            }
            else if (Settings::ImageSlot::CostFunction_NCC == fn.function) {
                int radius = 2;

                for (const auto& p : fn.parameters) {
                    if (p.first == "radius") {
                        radius = str_to_num<int>("NCCFunction", p.first, p.second);
                    }
                    else {
                        throw std::invalid_argument("[WebGPU] NCCFunction: unrecognised parameter "
                                                    "'" + p.first + "' with value '" + p.second + "'");
                    }
                }

                function = make_unique<WebGpuCostFunction_NCC>(fixed, moving, radius);
            }
            else {
                FATAL() << "[WebGPU] Unsupported cost function (slot: " << i << ")";
            }

            if (has_level(_fixed_mask_pyramid, level)) {
                function->set_fixed_mask(_fixed_mask_pyramid[level]);
            }
            if (has_level(_moving_mask_pyramid, level)) {
                function->set_moving_mask(_moving_mask_pyramid[level]);
            }

            unary_fn.add_function(function, fn.weight);
        }
    }

    if (_fixed_landmarks.size() > 0 && level >= _settings.landmarks_stop_level) {
        ASSERT(_fixed_landmarks.size() == _moving_landmarks.size());

        FunctionPtr f = make_unique<WebGpuCostFunction_Landmarks>(
                _fixed_landmarks,
                _moving_landmarks,
                _settings.levels[level].landmarks_decay
                );
        unary_fn.add_function(f, _settings.levels[level].landmarks_weight);
    }

    if (has_level(_constraints_mask_pyramid, level)) {
        FunctionPtr f = make_unique<WebGpuCostFunction_SoftConstraints>(
                _constraints_mask_pyramid[level],
                _constraints_pyramid[level]
        );
        unary_fn.add_function(f, _settings.levels[level].constraints_weight);
    }
}
void WebGpuRegistrationEngine::build_binary_function(int level, WebGpuBinaryFunction& binary_fn)
{
    binary_fn.set_fixed_spacing(_fixed_pyramids[0][level].spacing());
    binary_fn.set_regularization_weight(_settings.levels[level].regularization_weight);
    binary_fn.set_regularization_scale(_settings.levels[level].regularization_scale);
    binary_fn.set_regularization_exponent(_settings.levels[level].regularization_exponent);

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    if (has_level(_regularization_weight_map, level))
        binary_fn.set_weight_map(_regularization_weight_map[level]);
#endif
}

WebGpuRegistrationEngine::WebGpuRegistrationEngine(const Settings& settings) :
    _settings(settings)
{
    stk::webgpu::initialize();

    // Guess from settings, it will be resized later if too small
    _fixed_pyramids.resize(settings.image_slots.size());
    _moving_pyramids.resize(settings.image_slots.size());
}
WebGpuRegistrationEngine::~WebGpuRegistrationEngine()
{
}

void WebGpuRegistrationEngine::set_initial_displacement_field(const stk::Volume& def)
{
    // GPU prefers float4 over float3
    ASSERT(def.voxel_type() == stk::Type_Float4 || def.voxel_type() == stk::Type_Float3);
    ASSERT(_settings.num_pyramid_levels);

    // Upload
    stk::WebGpuVolume gpu_def;
    if (def.voxel_type() == stk::Type_Float3)
        gpu_def = stk::WebGpuVolume(volume_float3_to_float4(def));
    else
        gpu_def = stk::WebGpuVolume(def);

    _deformation_pyramid = build_pyramid(
        gpu_def, filters::webgpu::downsample_vectorfield_by_2);
}
void WebGpuRegistrationEngine::set_affine_transform(const AffineTransform& affine_transform)
{
    _affine_transform = affine_transform;
}
void WebGpuRegistrationEngine::set_image_pair(
        int i,
        const stk::Volume& fixed,
        const stk::Volume& moving)
{
    ASSERT(_fixed_pyramids.size() == _moving_pyramids.size());
    FATAL_IF(fixed.voxel_type() != stk::Type_Float ||
             moving.voxel_type() != stk::Type_Float)
        << "Unsupported format";

    if (i >= (int) _fixed_pyramids.size()) {
        _fixed_pyramids.resize(i + 1);
        _moving_pyramids.resize(i + 1);
    }

    auto downsample_fn = filters::webgpu::downsample_volume_by_2;

    stk::WebGpuVolume gpu_fixed;
    stk::WebGpuVolume gpu_moving;
    {
        PROFILER_SCOPE("upload_image_pair", 0xFF532439);
        gpu_fixed = stk::WebGpuVolume(fixed);
        gpu_moving = stk::WebGpuVolume(moving);
    }

    _fixed_pyramids[i] = build_pyramid(gpu_fixed, downsample_fn);
    _moving_pyramids[i] = build_pyramid(gpu_moving, downsample_fn);
}

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
void WebGpuRegistrationEngine::set_regularization_weight_map(const stk::Volume& map)
{
    FATAL_IF(map.voxel_type() != stk::Type_Float) << "Unsupported format";

    _regularization_weight_map = build_pyramid(
        stk::WebGpuVolume(map), filters::webgpu::downsample_volume_by_2);
}
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

void WebGpuRegistrationEngine::set_landmarks(const std::vector<float3>& fixed_landmarks,
                                             const std::vector<float3>& moving_landmarks)
{
    ASSERT(fixed_landmarks.size() == moving_landmarks.size());
    _fixed_landmarks = fixed_landmarks;
    _moving_landmarks = moving_landmarks;
}

void WebGpuRegistrationEngine::set_fixed_mask(const stk::VolumeFloat& fixed_mask)
{
    _fixed_mask_pyramid = build_pyramid(
        stk::WebGpuVolume(fixed_mask), filters::webgpu::downsample_volume_by_2);
}

void WebGpuRegistrationEngine::set_moving_mask(const stk::VolumeFloat& moving_mask)
{
    _moving_mask_pyramid = build_pyramid(
        stk::WebGpuVolume(moving_mask), filters::webgpu::downsample_volume_by_2);
}

void WebGpuRegistrationEngine::set_voxel_constraints(const stk::VolumeUChar& mask,
                                                     const stk::VolumeFloat3& values)
{
    stk::WebGpuVolume gpu_mask(volume_uchar_to_float(mask));
    stk::WebGpuVolume gpu_values;
    if (values.voxel_type() == stk::Type_Float3)
        gpu_values = stk::WebGpuVolume(volume_float3_to_float4(values));
    else
        gpu_values = stk::WebGpuVolume(values);

    webgpu_voxel_constraints::build_pyramids(
        gpu_mask,
        gpu_values,
        _settings.num_pyramid_levels,
        _constraints_mask_pyramid,
        _constraints_pyramid
    );
}

stk::Volume WebGpuRegistrationEngine::execute()
{
    LOG(Info) << "Running WebGPU supported registration";

    FATAL_IF(_fixed_pyramids.empty() || !has_level(_fixed_pyramids[0], 0))
        << "No image pair set";

    if (!has_level(_deformation_pyramid, 0)) {
        // No initial deformation, create a field with all zeros

        stk::WebGpuVolume base = _fixed_pyramids[0][0];

        stk::VolumeFloat4 initial(base.size(), float4{0, 0, 0, 0});
        initial.set_origin(base.origin());
        initial.set_spacing(base.spacing());
        initial.set_direction(base.direction());

        set_initial_displacement_field(initial);
    }

    for (int l = _settings.num_pyramid_levels-1; l >= 0; --l) {
        WebGpuDisplacementField df(_deformation_pyramid[l], _affine_transform);

        std::vector<int3> neighborhood = determine_neighborhood(l);

        if (l >= _settings.pyramid_stop_level) {
            LOG(Info) << "Performing registration level " << l;

            if (has_level(_constraints_mask_pyramid, l)) {
                webgpu_voxel_constraints::constraint_displacement_field(
                    df,
                    _constraints_mask_pyramid[l],
                    _constraints_pyramid[l]
                );
            }

            WebGpuUnaryFunction unary_fn;
            build_unary_function(l, unary_fn);

            WebGpuBinaryFunction binary_fn;
            build_binary_function(l, binary_fn);

            if (_settings.solver == Settings::Solver_ICM) {
                WebGpuGraphCutOptimizer<ICMSolver<double>> optimizer(
                    neighborhood,
                    _settings.levels[l],
                    _settings.update_rule,
                    unary_fn,
                    binary_fn,
                    df
                );
                optimizer.execute();
            }
#if defined(DF_ENABLE_GCO)
            else if (_settings.solver == Settings::Solver_GCO) {
                WebGpuGraphCutOptimizer<GCOSolver<double>> optimizer(
                    neighborhood,
                    _settings.levels[l],
                    _settings.update_rule,
                    unary_fn,
                    binary_fn,
                    df
                );
                optimizer.execute();
            }
#endif
#if defined(DF_ENABLE_GRIDCUT)
            else if (_settings.solver == Settings::Solver_GridCut) {
                WebGpuGraphCutOptimizer<GridCutSolver<double>> optimizer(
                    neighborhood,
                    _settings.levels[l],
                    _settings.update_rule,
                    unary_fn,
                    binary_fn,
                    df
                );
                optimizer.execute();
            }
#endif
            stk::webgpu::check_errors();
        }
        else {
            LOG(Info) << "Skipping level " << l;
        }

        stk::WebGpuVolume vf = df.volume();
        if (l != 0) {
            dim3 upsampled_dims = _deformation_pyramid[l - 1].size();
            _deformation_pyramid[l - 1] =
                filters::webgpu::upsample_vectorfield(vf, upsampled_dims);
        }
        else {
            _deformation_pyramid[0] = vf;
        }
    }

    stk::Volume out = volume_float4_to_float3(
        webgpu::compute_displacement_field(
            _deformation_pyramid[0],
            _affine_transform
        ).download()
    );
    stk::webgpu::check_errors();
    return out;
}
std::vector<int3> WebGpuRegistrationEngine::determine_neighborhood(int level) const
{
    // Identical to RegistrationEngine::determine_neighborhood with the exception
    //  of working on WebGPU volumes

    std::vector<int3> neighborhood;

    dim3 dim_size {0, 0, 0};

    for (int i = 0; i < (int) _fixed_pyramids.size(); ++i) {
        dim3 fixed_size {0, 0, 0};
        dim3 moving_size {0, 0, 0};

        if (has_level(_fixed_pyramids[i], level))
            fixed_size = _fixed_pyramids[i][level].size();
        if (has_level(_moving_pyramids[i], level))
            moving_size = _moving_pyramids[i][level].size();

        dim_size = {
            std::max(dim_size.x, std::max(fixed_size.x, moving_size.x)),
            std::max(dim_size.y, std::max(fixed_size.y, moving_size.y)),
            std::max(dim_size.z, std::max(fixed_size.z, moving_size.z))
        };
    }

    if (dim_size.x > 1) {
        neighborhood.push_back({1, 0, 0});
        neighborhood.push_back({-1, 0, 0});
    }

    if (dim_size.y > 1) {
        neighborhood.push_back({0, 1, 0});
        neighborhood.push_back({0, -1, 0});
    }

    if (dim_size.z > 1) {
        neighborhood.push_back({0, 0, 1});
        neighborhood.push_back({0, 0, -1});
    }

    return neighborhood;
}

#endif // DF_USE_WEBGPU
