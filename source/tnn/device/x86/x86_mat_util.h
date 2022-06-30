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

#ifndef TNN_SOURCE_TNN_DEVICE_X86_X86_MAT_UTIL_H_
#define TNN_SOURCE_TNN_DEVICE_X86_X86_MAT_UTIL_H_

#include <string.h>
#include <cstdlib>

#include "tnn/core/blob.h"
#include "tnn/core/macro.h"
#include "tnn/utils/bfp16.h"

namespace TNN_NS {
namespace x86 {

#define GET_OFFSET_PTR(ptr, offset) (reinterpret_cast<int8_t*>(ptr) + offset)

void MatMemcpy2D(void* src, void* dst, int width, int height, int src_stride, int dst_stride);
void MatMemcpy2DWithPadding(void* src, void* dst, int width, int height, int src_stride, int dst_stride, int top,
                            int bottom, int left, int right, uint8_t pad_val);

// color convert
void NV12ToBGR(const unsigned char* nv12, unsigned char* bgr, int height, int width);
void NV21ToBGR(const unsigned char* nv21, unsigned char* bgr, int height, int width);
void NV12ToBGRA(const unsigned char* nv12, unsigned char* bgra, int height, int width);
void NV21ToBGRA(const unsigned char* nv21, unsigned char* bgra, int height, int width);

void BGRToGray(const unsigned char* bgr, unsigned char* gray, int height, int width);
void BGRAToGray(const unsigned char* bgra, unsigned char* gray, int height, int width);
void RGBToGray(const unsigned char* rgb, unsigned char* gray, int height, int width);
void RGBAToGray(const unsigned char* rgba, unsigned char* gray, int height, int width);

// resize
void ResizeBilinearC1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void ResizeBilinearC2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void ResizeBilinearC3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void ResizeBilinearC4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void ResizeBilinearYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);

void ResizeNearestC1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void ResizeNearestC2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void ResizeNearestC3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void ResizeNearestC4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void ResizeNearestYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);

// warp affine
void WarpAffineBilinearC1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                          const float (*transform)[3], const float border_val = 0.0);
void WarpAffineBilinearC2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                          const float (*transform)[3], const float border_val = 0.0);
void WarpAffineBilinearC3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                          const float (*transform)[3], const float border_val = 0.0);
void WarpAffineBilinearC4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                          const float (*transform)[3], const float border_val = 0.0);
void WarpAffineBilinearYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                                const float (*transform)[3], const float border_val = 0.0);

void WarpAffineNearestC1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                         const float (*transform)[3], const float border_val = 0.0);
void WarpAffineNearestC2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                         const float (*transform)[3], const float border_val = 0.0);
void WarpAffineNearestC3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                         const float (*transform)[3], const float border_val = 0.0);
void WarpAffineNearestC4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                         const float (*transform)[3], const float border_val = 0.0);
void WarpAffineNearestYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                               const float (*transform)[3], const float border_val = 0.0);

}  // namespace x86
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_X86_X86_MAT_UTIL_H_
