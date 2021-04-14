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

#ifndef TNN_ARM_WINOGRAD_FUNCTION_H_
#define TNN_ARM_WINOGRAD_FUNCTION_H_

#include <stdint.h>
#include <stdio.h>
#include "tnn/core/macro.h"
#include "tnn/device/arm/arm_common.h"

namespace TNN_NS {

typedef void (*SrcTransformFunc)(const void* src, void* dst,
                            int w_stride, int h_stride);
typedef void (*DstTransformFunc)(const void* src, void* dst,
                            int w_stride, int h_stride, int ey);

void WeightTransform4x4(const float *src, float *dst, int kernel_size, int in_channel, int out_channel);
void WeightTransform6x6(const float *src, float *dst, int kernel_size, int in_channel, int out_channel);

void SrcTransform4x4(const float *src, float *dst, int src_stride, int dst_stride);
void SrcTransform6x6(const float *src, float *dst, int src_stride, int dst_stride);

void DstTransform4x2(const float *src, float *dst, int src_stride, int dst_stride);
void DstTransform6x4(const float *src, float *dst, int src_stride, int dst_stride);

void SrcTransformInOne4x4Float(const void *src, void *dst, int w_stride, int h_stride);
void DstTransformInOne4x2Float(const void *src, void *dst, int w_stride, int h_stride, int ey);
void SrcTransformInOne6x6Float(const void *src, void *dst, int w_stride, int h_stride);
void DstTransformInOne6x4Float(const void *src, void *dst, int w_stride, int h_stride, int ey);

void SrcTransformInOne4x4BFP16(const void *src, void *dst, int w_stride, int h_stride);
void DstTransformInOne4x2BFP16(const void *src, void *dst, int w_stride, int h_stride, int ey);
void SrcTransformInOne6x6BFP16(const void *src, void *dst, int w_stride, int h_stride);
void DstTransformInOne6x4BFP16(const void *src, void *dst, int w_stride, int h_stride, int ey);

#if TNN_ARM82
void WeightTransformHalf4x4(const float *src, float *dst, int kernel_size, int in_channel, int out_channel);
void WeightTransformHalf6x6(const float *src, float *dst, int kernel_size, int in_channel, int out_channel);

void SrcTransform4x4(const fp16_t *src, fp16_t *dst, int src_stride, int dst_stride);
void SrcTransform6x6(const fp16_t *src, fp16_t *dst, int src_stride, int dst_stride);

void DstTransform4x2(const fp16_t *src, fp16_t *dst, int src_stride, int dst_stride);
void DstTransform6x4(const fp16_t *src, fp16_t *dst, int src_stride, int dst_stride);

void SrcTransformInOne4x4Fp16(const void *src, void *dst, int w_stride, int h_stride);
void DstTransformInOne4x2Fp16(const void *src, void *dst, int w_stride, int h_stride, int ey);
void SrcTransformInOne6x6Fp16(const void *src, void *dst, int w_stride, int h_stride);
void DstTransformInOne6x4Fp16(const void *src, void *dst, int w_stride, int h_stride, int ey);
#endif

}  // namespace TNN_NS

#endif /* WinogradOptFunction_hpp */
