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

#ifndef TNN_ARM_COMPUTE_H_
#define TNN_ARM_COMPUTE_H_

#include <float.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>

#include "tnn/core/macro.h"
#include "tnn/utils/bfp16.h"

namespace TNN_NS {
struct ArmKernelParam {
    long ic_r4;
    long ih;
    long iw;
    long oc_r4;
    long oh;
    long ow;
    void* fil_ptr;
    float* scale;
    void* bias;
};

typedef void (*PostFunc)(void* dst, const float* bias, long area, long oc4);
typedef void (*ConvDwSliceFunc)(void* dst_z, void** cache_line, const void* weight_z, long dst_width);

template <typename T>
void PostAddBias(void* dst, const float* bias, long area, long oc4);

template <typename T>
void PostAddBiasRelu(void* dst, const float* bias, long area, long oc4);

template <typename T>
void PostAddBiasRelu6(void* dst, const float* bias, long area, long oc4);

template <typename T>
void PostClap(void* dst, long size4, float val);

template <typename T>
void DepthwiseUnit(T* dst, const T* src, const float* weight, long fw, long fh, long weight_y_step, long dilateX_step,
                   long dilateY_step);
template <typename T>
void DepthwiseConv(T* dst, const T* src, const float* weight, long width, long src_w_setup, long fw, long fh,
                   long dilateX_step, long dilateY_step, long height, long srcHStep, long dstHStep);
template <typename T>
void DepthwiseConv3x3(T* dst, const T* src, const float* weight, long width, long src_w_setup, long fw, long fh,
                      long dilateX_step, long dilateY_step, long height, long srcHStep, long dstHStep);

template <typename T>
void DepthwiseUnitDeconv(const T* dst, T* src, const float* weight, long fw, long fh, long weight_y_step,
                         long dilateX_step, long dilateY_step);

template <typename T>
void DepthwiseDeconv(const T* dst, T* src, const float* weight, long width, long src_w_setup, long fw, long fh,
                     long dilateX_step, long dilateY_step);

template <typename T>
void MaxPooling(const T* src, long inputWidth, long inputHeight, T* dst, long outputWidth, long outputHeight,
                long kernelWidth, long kernelHeight, long strideWidth, long strideHeight, long padWidth, long padHeight,
                long l, long r, long t, long b);

template <typename T>
void AvgPooling(const T* src, long inputWidth, long inputHeight, T* dst, long outputWidth, long outputHeight,
                long kernelWidth, long kernelHeight, long strideWidth, long strideHeight, long padWidth,
                long padHeight);

template <typename T>
void ConvCommonO4(T* dst, const T* src, const float* weight, long width, long src_w_step, long src_depth_quad,
                  long src_depth_step, long fw, long fh, long dilate_x_step, long dilate_y_step);

template <typename Tin, typename Tout>
void FloatConvert(const Tin* src, Tout* dst, long area_quad);

#ifdef __cplusplus
extern "C" {
#endif
void ConvFloatO4(float* dst, const float* src, const float* weight, long width, long src_w_step, long src_depth_quad,
                 long src_depth_step, long fw, long fh, long dilate_x_step, long dilate_y_step);
void ConvBfp16O4(bfp16_t* dst, const bfp16_t* src, const float* weight, long width, long src_w_step,
                 long src_depth_quad, long src_depth_step, long fw, long fh, long dilate_x_step, long dilate_y_step);

void DeconvFloatO4(float* dst, const float* src, const float* weight, long width, long dst_w_step, long src_depth_quad,
                   long src_depth_step, long fw, long fh, long dilateX_step, long dilateY_step);
void DeconvBfp16O4(bfp16_t* dst, const bfp16_t* src, const float* weight, long width, long dst_w_step,
                   long src_depth_quad, long src_depth_step, long fw, long fh, long dilateX_step, long dilateY_step);

void GemmFloatSlidewC3(float* dst, const float* src, const float* weight, long width, long src_w_setup, long fw,
                       long fh, long dilateX_step, long dilateY_step);
void GemmBfp16SlidewC3(bfp16_t* dst, const bfp16_t* src, const float* weight, long width, long src_w_setup, long fw,
                       long fh, long dilateX_step, long dilateY_step);

void GEMM_FLOAT_N8(float* dst, const float* src, const float* weight, long src_depth_quad, long dst_step,
                   long dst_depth_quad, long width, float* bias, int64_t relu);
void GEMM_BFP16_N8(bfp16_t* dst, const bfp16_t* src, const float* weight, long src_depth_quad, long dst_step,
                   long dst_depth_quad, long width, float* bias, int64_t relu);

void GEMM_FLOAT_N4(float* dst, const float* src, const float* weight, long src_depth_quad, long dst_step,
                   long dst_depth_quad, long width, float* bias, long relu);
void GEMM_BFP16_N4(bfp16_t* dst, const bfp16_t* src, const float* weight, long src_depth_quad, long dst_step,
                   long dst_depth_quad, long width, float* bias, long relu);

void ConvDw3x3FloatSlideW(void* dst_z, void** cache_line, const void* weight_z, long dst_width);
void ConvDw3x3Bfp16SlideW(void* dst_z, void** cache_line, const void* weight_z, long dst_width);
void ConvDw5x5FloatSlideW(void* dst_z, void** cache_line, const void* weight_z, long dst_width);
void ConvDw5x5Bfp16SlideW(void* dst_z, void** cache_line, const void* weight_z, long dst_width);

#ifdef __cplusplus
}
#endif
}  // namespace TNN_NS
#endif
