#pragma once

//#define DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
#define DF_ENABLE_REGULARIZATION_WEIGHT_MAP

// Use the SSE version of linear_at<float>
// Does not make any real difference in performance on msvc2017
// #define DF_ENABLE_SSE_LINEAR_AT

#ifdef _DEBUG
    #define DF_OUTPUT_VOLUME_ENERGY
    //#define DF_OUTPUT_DEBUG_VOLUMES
#endif

// GPU supported registration is available through either CUDA or WebGPU
#if defined(DF_USE_CUDA) || defined(DF_USE_WEBGPU)
    #define DF_USE_GPU
#endif
