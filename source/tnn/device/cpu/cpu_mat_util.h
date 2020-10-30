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

#ifndef TNN_SOURCE_TNN_DEVICE_CPU_CPU_MAT_UTIL_H_
#define TNN_SOURCE_TNN_DEVICE_CPU_CPU_MAT_UTIL_H_
#include <string.h>
#include <sys/time.h>
#include <cstdlib>

#include "tnn/core/blob.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

#define GET_OFFSET_PTR(ptr, offset) (reinterpret_cast<int8_t*>(ptr) + offset)

void WarpAffineBilinear(const uint8_t* src, int src_w, int src_h, int channel, uint8_t* dst, int dst_w, int dst_h,
                        const float (*transform)[3], const float border_val = 0.0);
void WarpAffineNearest(const uint8_t* src, int src_w, int src_h, int channel, uint8_t* dst, int dst_w, int dst_h,
                       const float (*transform)[3], const float border_val = 0.0);
void ResizeBilinear(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h, int channel);
void ResizeNearest(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h, int channel);
void BGROrBGRAToGray(const uint8_t* src, uint8_t* dst, int h, int w, int channel);
void YUVToBGR(const unsigned char* yuv, unsigned char* bgr, int h, int w, bool is_nv12);
void YUVToBGRA(const unsigned char* yuv, unsigned char* bgra, int h, int w, bool is_nv12);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_CPU_CPU_MAT_UTIL_H_
