// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_INCLUDE_CUDA_MACRO_H_
#define TNN_INCLUDE_CUDA_MACRO_H_

#include <sstream>

#include "core/macro.h"

// Interface visibility
#define PUBLIC __attribute__((visibility("default")))

// IEEE 754
#ifndef FLT_MIN
#define FLT_MIN 1.175494351e-38F
#define FLT_MAX 3.402823466e+38F
#define FLT_EPSILON 1.192092896e-07F
#endif

#define FatalError(s)                                                          \
    {                                                                          \
        std::stringstream _where, _message;                                    \
        _where << __FILE__ << ':' << __LINE__;                                 \
        _message << std::string(s) + "\n"                                      \
                 << __FILE__ << ':' << __LINE__ << "\nAborting... \n";         \
        LOGE("%s", _message.str().c_str());                                    \
        cudaDeviceReset();                                                     \
        exit(EXIT_FAILURE);                                                    \
    }

#define CUDNN_CHECK(status)                                                    \
    {                                                                          \
        std::stringstream _error;                                              \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            _error << "CUDNN failure: " << cudnnGetErrorString(status);        \
            FatalError(_error.str());                                          \
        }                                                                      \
    }

#define CUDA_CHECK(status)                                                     \
    {                                                                          \
        std::stringstream _error;                                              \
        if (status != cudaSuccess) {                                           \
            _error << "Cuda failure: " << cudaGetErrorName(status) << " "      \
                   << cudaGetErrorString(status);                              \
            FatalError(_error.str());                                          \
        }                                                                      \
    }

#define CUBLAS_CHECK(status)                                                   \
    {                                                                          \
        std::stringstream _error;                                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            _error << "Cublas failure: "                                       \
                   << " " << status;                                           \
            FatalError(_error.str());                                          \
        }                                                                      \
    }

#define CUDA_KERNEL_LOOP(i, n)                                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);               \
         i += blockDim.x * gridDim.x)

namespace TNN_NS {

const int RPD_CUDA_NUM_THREADS = 512;

inline int RPD_GET_BLOCKS(const int N) {
    return (N + RPD_CUDA_NUM_THREADS - 1) / RPD_CUDA_NUM_THREADS;
}

}  // namespace TNN_NS

#endif
