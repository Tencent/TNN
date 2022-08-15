// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the 
// specific language governing permissions and limitations under the License.

#ifndef TNN_SOURCE_TNN_DEVICE_CUDA_CUDA_MACRO_H_
#define TNN_SOURCE_TNN_DEVICE_CUDA_CUDA_MACRO_H_

#include <sstream>

#include "tnn/core/macro.h"

namespace TNN_NS {

#define FatalError(err) {                                                  \
    std::stringstream _where, _message;                                    \
        _where << __FILE__ << ':' << __LINE__;                             \
        _message << std::string(err) + "\n"                                \
                 << __FILE__ << ':' << __LINE__ << "\nAborting... \n";     \
        LOGE("%s", _message.str().c_str());                                \
        exit(EXIT_FAILURE);                                                \
}

#define CUDA_CHECK(status) {                                               \
    std::stringstream _error;                                              \
    if (cudaSuccess != status) {                                           \
        _error << "Cuda failure: " << cudaGetErrorName(status) << " "      \
               << cudaGetErrorString(status);                              \
        FatalError(_error.str());                                          \
    }                                                                      \
}

#define CUDNN_CHECK(status)                                                    \
    {                                                                          \
        std::stringstream _error;                                              \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            _error << "CUDNN failure: " << cudnnGetErrorString(status);        \
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


#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

#define TNN_CUDA_NUM_THREADS 128

#define TNN_CUDA_MAX_GRID_DIM 65535

inline int TNN_CUDA_GET_BLOCKS(const int N) {
    return (N + TNN_CUDA_NUM_THREADS - 1) / TNN_CUDA_NUM_THREADS;
}

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_DEVICE_CUDA_CUDA_MACRO_H_
