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

#include "tnn/device/x86/x86_mat_util.h"

#include <algorithm>
#include <type_traits>
#include "stdlib.h"

#ifdef TNN_USE_NEON
#include <x86_neon.h>
#endif

#include "tnn/core/macro.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/mat_converter_utils.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

static inline void* x86Malloc(size_t size) {
    return _mm_malloc(size, 32);
}

static inline void x86Free(void* ptr) {
    _mm_free(ptr);
}

#define SATURATE_CAST_UCHAR(X)                                                                                         \
    (unsigned char)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), (int)0), (int)UCHAR_MAX)
#define SATURATE_CAST_SHORT(X)                                                                                         \
    (short)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), (int)SHRT_MIN), (int)SHRT_MAX)
#define SATURATE_CAST_INT(X)                                                                                           \
    (int)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), (int)INT_MIN), (int)INT_MAX)

void MatMemcpy2D(void* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    auto src_ptr = reinterpret_cast<uint8_t*>(src);
    auto dst_ptr = reinterpret_cast<uint8_t*>(dst);

    for (int h = 0; h < height; h++) {
        memcpy(dst_ptr, src_ptr, width);
        src_ptr += src_stride;
        dst_ptr += dst_stride;
    }
}

void MatMemcpy2DWithPadding(void* src, void* dst, int width, int height, int src_stride, int dst_stride, int top,
                            int bottom, int left, int right, uint8_t pad_val) {
    auto src_ptr = reinterpret_cast<uint8_t*>(src);
    auto dst_ptr = reinterpret_cast<uint8_t*>(dst);

    int top_plane = top * dst_stride;
    memset(dst_ptr, pad_val, top_plane);
    dst_ptr += top_plane;

    for (int h = 0; h < height; h++) {
        memset(dst_ptr, pad_val, left);
        dst_ptr += left;
        memcpy(dst_ptr, src_ptr, width);
        src_ptr += src_stride;
        dst_ptr += width;
        memset(dst_ptr, pad_val, right);
        dst_ptr += right;
    }

    int bottom_plane = bottom * dst_stride;
    memset(dst_ptr, pad_val, bottom_plane);
}

// color convert
void NV12ToBGR(const unsigned char* nv12, unsigned char* bgr, int height, int width) {}
void NV21ToBGR(const unsigned char* nv21, unsigned char* bgr, int height, int width) {}
void NV12ToBGRA(const unsigned char* nv12, unsigned char* bgra, int height, int width) {}
void NV21ToBGRA(const unsigned char* nv21, unsigned char* bgra, int height, int width) {}

void BGRToGray(const unsigned char* bgr, unsigned char* gray, int height, int width) {}
void BGRAToGray(const unsigned char* bgra, unsigned char* gray, int height, int width) {}
void RGBToGray(const unsigned char* rgb, unsigned char* gray, int height, int width) {}
void RGBAToGray(const unsigned char* rgba, unsigned char* gray, int height, int width) {}

// resize
void ResizeBilinearC1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {}
void ResizeBilinearC2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {}
void ResizeBilinearC3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {}
void ResizeBilinearC4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {}
void ResizeBilinearYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {}

void ResizeNearestC1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {}
void ResizeNearestC2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {}
void ResizeNearestC3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {}
void ResizeNearestC4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {}
void ResizeNearestYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {}

// warp affine
void WarpAffineBilinearC1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                          const float (*transform)[3], const float border_val) {}
void WarpAffineBilinearC2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                          const float (*transform)[3], const float border_val) {}
void WarpAffineBilinearC3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                          const float (*transform)[3], const float border_val) {}
void WarpAffineBilinearC4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                          const float (*transform)[3], const float border_val) {}
void WarpAffineBilinearYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                                const float (*transform)[3], const float border_val) {}

void WarpAffineNearestC1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                         const float (*transform)[3], const float border_val) {}
void WarpAffineNearestC2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                         const float (*transform)[3], const float border_val) {}
void WarpAffineNearestC3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                         const float (*transform)[3], const float border_val) {}
void WarpAffineNearestC4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                         const float (*transform)[3], const float border_val) {}
void WarpAffineNearestYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h,
                               const float (*transform)[3], const float border_val) {}

}  // namespace TNN_NS
