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

#ifndef TNN_ARM_MAT_UTIL_H_
#define TNN_ARM_MAT_UTIL_H_

#include <string.h>
#include <sys/time.h>
#include <cstdlib>

#include "tnn/core/blob.h"
#include "tnn/core/macro.h"
#include "tnn/utils/bfp16.h"

namespace TNN_NS {

#define GET_OFFSET_PTR(ptr, offset) (reinterpret_cast<int8_t*>(ptr) + offset)

void mat_memcpy_2d(void* src, void* dst, int width, int height, int src_stride, int dst_stride);

// resize
void resize_bilinear_c1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void resize_bilinear_c2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void resize_bilinear_c3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void resize_bilinear_c4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void resize_bilinear_yuv420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);

void resize_nearest_c1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void resize_nearest_c2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void resize_nearest_c3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void resize_nearest_c4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);
void resize_nearest_yuv420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h);

// warp affine
void warpaffine_bilinear_c1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                            const float (*transform)[3], const float border_val = 0.0);
void warpaffine_bilinear_c3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                            const float (*transform)[3], const float border_val = 0.0);

}  // namespace TNN_NS

#endif
