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

#ifndef TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_STRIDED_SLICE_LAYER_ACC_KERNEL_CUH_
#define TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_STRIDED_SLICE_LAYER_ACC_KERNEL_CUH_

#include "tnn/device/cuda/acc/cuda_layer_acc.h"

namespace TNN_NS {

Status RunStrideSlice(int size, const float * src_data, int input_c, int input_h,
        int input_w, const int* begin, const int* strides, float* dst_data,
        int output_c, int output_h, int output_w, int div_c, int div_n, cudaStream_t stream);

Status RunStrideSlice(int size, const float * src_data, int input_c, int input_d, int input_h,
        int input_w, const int* begin, const int* strides, float* dst_data,
        int output_c, int output_d, int output_h, int output_w, int div_d, int div_c, int div_n, cudaStream_t stream);

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_STRIDED_SLICE_LAYER_ACC_KERNEL_CUH_
