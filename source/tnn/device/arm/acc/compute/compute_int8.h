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
#include "tnn/device/arm/acc/compute/compute.h"
#include "tnn/utils/bfp16.h"

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
struct Q8GemmContext {
    int32_t k;
    int32_t k_stride;
    int32_t n;
    int32_t n_stride;
    const int8_t* a;
    int32_t a_stride;
    const int8_t* packed_w;
    int8_t* c;
    int32_t c_stride;
    float* scales;
    int relu;
};

typedef void (*GemmInt8N8Func)(long mr, long nr, long k, const int8_t* a, long a_stride, const void* w, int8_t* c,
                               long c_stride, const float* scales, long);

void ComputeQ8Gemm(const Q8GemmContext* context, int32_t range_k, int32_t range_l, int32_t tile_k, int32_t tile_l);

void DepthwiseI8Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, long fw, long fh,
                     long weight_y_step, long src_y_step, const float* scale, long dst_depth);

void DepthwiseI8General(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
                        long src_y_step, long src_w_step, long dst_depth, long fw, long fh, const float* scale_z);

void DepthwiseI8K3(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
                   long src_y_step, long src_w_step, long dst_depth, long fw, long fh, const float* scale_z);

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
