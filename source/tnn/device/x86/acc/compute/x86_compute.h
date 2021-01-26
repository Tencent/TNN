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
#include "tnn/device/x86/acc/x86_reduce_op_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/device/x86/acc/Float4.h"
#include "tnn/device/x86/acc/Float8.h"

namespace TNN_NS {

// @brief store by row
Status X86_IM2COL(float *src, int channel, int height, int width, int kernelh, int kernelw, 
                  int padh, int padw, int strideh, int stridew, int dilationh, int dilationw, float *dst);

// @brief C = A * B with B tranposed, (m * k) * (k * n), NAIVE
Status X86_matrixMul(int m, int n, int k, const float *A, const float *B, float *C, 
                     int has_bias = 0, const float *bias = nullptr, int activation_type = ActivationType_None);

Status X86_MAX_POOLING(float *input, float *output, DimsVector input_dim, DimsVector output_dim,
                       int stride_h, int stride_w, int kernel_h, int kernel_w, int pad_h, int pad_w);

Status X86_AVERAGE_POOLING(float *input, float *output, DimsVector input_dim, DimsVector output_dim,
                           int stride_h, int stride_w, int kernel_h, int kernel_w, int pad_h, int pad_w);

Status X86_FMA(float *input, float *output, float *scale, float *bias,
               bool shared_channel, bool has_bias, DimsVector output_dim);

Status X86_REDUCE_CALCULATE(float *input, float *output, std::vector<int> axes,
                            DimsVector input_dim, DimsVector output_dim, X86ReduceOpType op_type);

template <class T, int pack_c>
void X86MaxPooling(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h, long l, long r, long t, long b);

template <class T, int pack_c>
void X86AvgPooling(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h);

template <int activation_type, typename VEC, int pack>
void DepthwiseConv(float* dst, const float* src, const float* weight, const float* bias, long width, long src_w_step, long fw, long fh,
                   long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep);

template <typename VEC, int pack>
void X86Sgemv(float* dst, const float* src, const float* weight, float *bias, DimsVector dims_input, DimsVector dims_output);
}   // namespace TNN_NS

#endif

