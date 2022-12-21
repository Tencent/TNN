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

#ifndef TNN_SOURCE_TNN_DEVICE_CUDA_CUDA_MAT_UTIL_CUH_
#define TNN_SOURCE_TNN_DEVICE_CUDA_CUDA_MAT_UTIL_CUH_
#include <string.h>

#include "tnn/core/blob.h"
#include "tnn/core/macro.h"
#include "tnn/utils/mat_utils.h"

namespace TNN_NS {

void ResizeBilinear(const uint8_t* src, uint8_t* dst, int batch, int src_w, int src_h, int dst_w, int dst_h, int channel);
void ResizeNearest(const uint8_t* src, uint8_t* dst, int batch, int src_w, int src_h, int w, int h, int channel);
void CropRGB(const uint8_t* src, uint8_t* dst, int batch, int channel, int src_width, int src_height, int dst_width,
        int dst_height, int width, int height, int top_left_x, int top_left_y);
void CropYUV(const uint8_t* src, uint8_t* dst, int batch, int src_width, int src_height, int dst_width, int dst_height,
        int width, int height, int top_left_x, int top_left_y);
void YUVToGRBA(const uint8_t* src, uint8_t* dst, int batch, int h, int w, int channel, bool is_nv12);
void BGRAToGRAY(const uint8_t* src, uint8_t* dst, int batch, int h, int w, int channel);
void CudaCopyMakeBorder(const uint8_t* src, uint8_t* dst, int batch, int src_width, int src_height, int dst_width,
        int dst_height, int channel, int top, int bottom, int left, int right, uint8_t pad_val);
void WarpAffineBilinear(const uint8_t* src, int batch, int channel, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
        const float (*transform)[3], const float border_val, BorderType border_type = BORDER_TYPE_CONSTANT, void* stream = nullptr);
void WarpAffineNearest(const uint8_t* src, int batch, int channel, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
        const float (*transform)[3], const float border_val, BorderType border_type = BORDER_TYPE_CONSTANT, void* stream = nullptr);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_CUDA_CUDA_MAT_UTIL_CUH_

