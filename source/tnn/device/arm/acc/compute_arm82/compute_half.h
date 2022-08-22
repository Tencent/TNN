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

#ifndef TNN_ARM_COMPUTE_HALF_H_
#define TNN_ARM_COMPUTE_HALF_H_

#include <float.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>

#include "tnn/core/macro.h"
#include "tnn/utils/half_utils_inner.h"

namespace TNN_NS {
namespace arm {

#if TNN_ARM82
// used for reformat
void HalfC8ToFloatC4(float* dst, const fp16_t* src, long batch, long channel, long hw);
void FloatC4ToHalfC8(fp16_t* dst, const float* src, long batch, long channel, long hw);
// used for blob converter
int PackNeon(fp16_t* dst, const fp16_t* src, size_t hw, size_t channel);
int PackNeonC3(fp16_t* dst, const float* src, size_t hw, size_t channel);
int PackNeonNHWC(fp16_t *dst, const fp16_t *src, size_t hw, size_t channel);
int UnpackNeonNHWC(fp16_t *dst, const fp16_t *src, size_t hw, size_t channel);
template <bool reverse_channel>
void BGRAToBlobImpl(const uint8_t* src, fp16_t* dst, const float* scale, const float* bias, int hw, int channel);
template <bool reverse_channel>
void BGRToBlobImpl(const uint8_t* src, fp16_t* dst, const float* scale, const float* bias, int hw);
void GrayToBlob(const uint8_t* src, fp16_t* dst, const float scale, const float bias, int hw);
template <bool reverse_channel>
void BlobToBGRAImpl(const fp16_t* src, uint8_t* dst, const float* scale, const float* bias, int hw, int channel);
template <bool reverse_channel>
void BlobToBGRImpl(const fp16_t* src, uint8_t* dst, const float* scale, const float* bias, int hw);

void GemmHalfPackA(int m, int n, int k, const fp16_t* a, fp16_t* pack_a, int lda, const fp16_t* b, int ldb, fp16_t* c,
                   int ldc);
void GemmFloatPackAB(int m, int n, int k, const fp16_t* a, fp16_t* pack_a, int lda, const fp16_t* b, fp16_t* pack_b, int ldb, fp16_t* c,
                   int ldc);
#endif

#ifdef TNN_ARM82_USE_NEON
#ifdef __cplusplus
extern "C" {
#endif
#endif

void Half2FloatKernel(float* dst, const fp16_t* src, const size_t length);
void Float2HalfKernel(fp16_t* dst, const float* src, const size_t length);
void GEMM_FP16_N8(fp16_t* dst, const fp16_t* src, const fp16_t* weight, long src_depth, long dst_step, long dst_depth,
                  long width, fp16_t* bias, long relu);
void GemmFp16SlidewC3(fp16_t* dst, const fp16_t* src, const fp16_t* weight, long width, long src_w_setup, long fw,
                      long fh, long dilateX_step, long dilateY_step);
void ConvDw3x3Fp16SlideW(void* dst_z, void** cache_line, const void* weight_z, long dst_width);
void DeconvFp16O8(fp16_t* dst, const fp16_t* src, const fp16_t* weight, long width, long dst_w_step,
                  long src_depth_quad, long src_depth_step, long fw, long fh, long dilateX_step, long dilateY_step);
void DeconvFp16O8C1(fp16_t* dst, const fp16_t* src, const fp16_t* weight, long width, long dst_w_step, long src_depth,
                    long src_depth_step, long fw, long fh, long dilateX_step, long dilateY_step);

#ifdef TNN_ARM82_USE_NEON
#ifdef __cplusplus
}
#endif
#endif

}  // namespace arm
}  // namespace TNN_NS

#endif  // TNN_ARM_COMPUTE_HALF_H_
