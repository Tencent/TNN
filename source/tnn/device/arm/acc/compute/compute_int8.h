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

#ifndef TNN_ARM_COMPUTE_INT8_H_
#define TNN_ARM_COMPUTE_INT8_H_

#include <float.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>

#include "tnn/core/macro.h"
#include "tnn/utils/bfp16.h"
#include "tnn/device/arm/acc/compute/compute.h"

namespace TNN_NS {

void MaxPoolingINT8(const int8_t* src, long inputWidth, long inputHeight, int8_t* dst, long outputWidth,
                    long outputHeight, long channel, long kernelWidth, long kernelHeight, long strideWidth,
                    long strideHeight, long padWidth, long padHeight);

void AvgPoolingINT8(const int8_t* src, long inputWidth, long inputHeight, int8_t* dst, long outputWidth,
                    long outputHeight, long channel, long kernelWidth, long kernelHeight, long strideWidth,
                    long strideHeight, long padWidth, long padHeight);

void MatrixAddInt8(int8_t* dst, const int8_t* A, const int8_t* B, float* dst_scale, const float* a_scale,
                   float* b_scale, long channel, long height, long width);

void Int8ToFloat(float* dst, const int8_t* src, const float* scale, long batch, long channel, long hw);

void FloatToInt8(int8_t* dst, const float* src, const float* scale, long batch, long channel, long hw);

#ifdef __cplusplus
extern "C" {
#endif
void DepthwiseConvI8(const int8_t* src, int8_t* dst, long dst_depth, long src_y_step, long dst_y_step, long dst_height,
                     long dst_width, long src_height, long src_width, long l, long r, long t, long b, long kernel,
                     const int8_t* weightPtr, const int32_t* biasPtr, const float* scalePtr, long stride, long pad,
                     ArmKernelParam* param);

void ReluInt8(int8_t* dst, const int8_t* src, long len);

void GemmInt8(int8_t* dst, const int8_t* src, int8_t* work_space, const int8_t* weight, const int32_t* bias,
              const float* scale, long src_depth_d8, long src_w_step, long dst_depth);

void GemvInt8(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, const float* scale, long ic_r8,
              long oc_r4);

#ifdef __cplusplus
}
#endif
}  // namespace TNN_NS
#endif
