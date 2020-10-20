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

#ifndef SOURCE_TNN_DEVICE_X86_ACC_COMPUTE_H_
#define SOURCE_TNN_DEVICE_X86_ACC_COMPUTE_H_

#include "tnn/core/common.h"
#include "tnn/core/blob.h"
#include "tnn/core/status.h"
#include "tnn/device/x86/acc/x86_binary_op_layer_acc.h"
#include "tnn/interpreter/layer_param.h"

namespace TNN_NS {

// binary
Status X86_BINARY_CALCULATE(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, 
                            Blob *output, std::shared_ptr<X86_BINARY_OP> op);

// @brief store by row
Status X86_IM2COL(float *src, int channel, int height, int width, int kernelh, int kernelw, 
                  int padh, int padw, int strideh, int stridew, int dilationh, int dilationw, float *dst);

// @brief C = A * B with B tranposed, (m * k) * (k * n), NAIVE
Status X86_matrixMul(int m, int n, int k, float *A, float *B, float *C, 
                     int has_bias = 0, float *bias = nullptr, int activation_type = ActivationType_None);

Status X86_MAX_POOLING(float *input, float *output, DimsVector input_dim, DimsVector output_dim,
                       int stride_h, int stride_w, int kernel_h, int kernel_w, int pad_h, int pad_w);

Status X86_AVERAGE_POOLING(float *input, float *output, DimsVector input_dim, DimsVector output_dim,
                           int stride_h, int stride_w, int kernel_h, int kernel_w, int pad_h, int pad_w);

}   // namespace TNN_NS

#endif

