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

#include "tnn/device/x86/x86_util.h"
#include "tnn/device/x86/x86_common.h"

#include <type_traits>

#include "tnn/core/macro.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {
namespace x86 {

#define _MM_TRANSPOSE4_LEFT(row0, row1, row2, row3) \
  __m128 tmp3, tmp2, tmp1, tmp0; \
  __m128 t0, t1, t2, t3; \
  tmp0 = _mm_unpacklo_ps((row0), (row1)); \
  tmp2 = _mm_unpacklo_ps((row2), (row3)); \
  tmp1 = _mm_unpackhi_ps((row0), (row1)); \
  tmp3 = _mm_unpackhi_ps((row2), (row3)); \
  t0 = _mm_movelh_ps(tmp0, tmp2); \
  t1 = _mm_movehl_ps(tmp2, tmp0); \
  t2 = _mm_movelh_ps(tmp1, tmp3); \
  t3 = _mm_movehl_ps(tmp3, tmp1);

template <int left_c>
inline void PackC4_Left(float *dst, const float *src, size_t hw, size_t src_hw_stride) {
    auto src0 = src;
    auto src1 = src + src_hw_stride;
    auto src2 = src + src_hw_stride * 2;

    int cur_hw = 0;
    __m128 v1 = _mm_setzero_ps();
    __m128 v2 = _mm_setzero_ps();
    __m128 v3 = _mm_setzero_ps();
    for (; cur_hw + 3 < hw; cur_hw += 4) {
        auto dst_hw = dst + cur_hw * 4;
        __m128 v0 = _mm_loadu_ps(src0 + cur_hw);
        if (left_c > 1) v1 = _mm_loadu_ps(src1 + cur_hw);
        if (left_c > 2) v2 = _mm_loadu_ps(src2 + cur_hw);
        _MM_TRANSPOSE4_LEFT(v0, v1, v2, v3);
        _mm_storeu_ps(dst_hw, t0);
        _mm_storeu_ps(dst_hw + 4, t1);
        _mm_storeu_ps(dst_hw + 8, t2);
        _mm_storeu_ps(dst_hw + 12, t3);
    }
    for (; cur_hw < hw; cur_hw++) {
        dst[cur_hw * 4 + 0] = src0[cur_hw];
        if (left_c > 1) {
            dst[cur_hw * 4 + 1] = src1[cur_hw];
        } else {
            dst[cur_hw * 4 + 1] = 0;
        }
        if (left_c > 2) {
            dst[cur_hw * 4 + 2] = src2[cur_hw];
        } else {
            dst[cur_hw * 4 + 2] = 0;
        }
        dst[cur_hw * 4 + 3] = 0;
    }
}

int PackC4(float *dst, const float *src, size_t hw, size_t src_hw_stride, size_t dst_hw_stride, size_t channel) {
    int c = 0;
    for (; c + 3 < channel; c += 4) {
        auto src0 = src + c * src_hw_stride;
        auto src1 = src0 + src_hw_stride;
        auto src2 = src0 + src_hw_stride * 2;
        auto src3 = src0 + src_hw_stride * 3;
        auto dst_c = dst + c * dst_hw_stride;
        int cur_hw = 0;
        for (; cur_hw + 3 < hw; cur_hw += 4) {
            auto dst_hw = dst_c + cur_hw * 4;
            __m128 v0 = _mm_loadu_ps(src0 + cur_hw);
            __m128 v1 = _mm_loadu_ps(src1 + cur_hw);
            __m128 v2 = _mm_loadu_ps(src2 + cur_hw);
            __m128 v3 = _mm_loadu_ps(src3 + cur_hw);
            _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
            _mm_storeu_ps(dst_hw, v0);
            _mm_storeu_ps(dst_hw + 4, v1);
            _mm_storeu_ps(dst_hw + 8, v2);
            _mm_storeu_ps(dst_hw + 12, v3);
        }
        for (; cur_hw < hw; cur_hw++) {
            dst_c[cur_hw * 4 + 0] = src0[cur_hw];
            dst_c[cur_hw * 4 + 1] = src1[cur_hw];
            dst_c[cur_hw * 4 + 2] = src2[cur_hw];
            dst_c[cur_hw * 4 + 3] = src3[cur_hw];
        }
    }
    int left_c = channel - c;
    if (left_c == 3) {
        PackC4_Left<3>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, src_hw_stride);
    } else if (left_c == 2) {
        PackC4_Left<2>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, src_hw_stride);
    } else if (left_c == 1) {
        PackC4_Left<1>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, src_hw_stride);
    }
    return 0;
}

#define _MM256_TRANSPOSE8(v0, v1, v2, v3, v4, v5, v6, v7) \
    __m256 t0 = _mm256_unpacklo_ps(v0, v1); \
    __m256 t1 = _mm256_unpackhi_ps(v0, v1); \
    __m256 t2 = _mm256_unpacklo_ps(v2, v3); \
    __m256 t3 = _mm256_unpackhi_ps(v2, v3); \
    __m256 t4 = _mm256_unpacklo_ps(v4, v5); \
    __m256 t5 = _mm256_unpackhi_ps(v4, v5); \
    __m256 t6 = _mm256_unpacklo_ps(v6, v7); \
    __m256 t7 = _mm256_unpackhi_ps(v6, v7); \
    __m256 v;                               \
    v  = _mm256_shuffle_ps(t0, t2, 0x4E);   \
    v0 = _mm256_blend_ps(t0, v, 0xCC);      \
    v1 = _mm256_blend_ps(t2, v, 0x33);      \
    v  = _mm256_shuffle_ps(t1, t3, 0x4E);   \
    v2 = _mm256_blend_ps(t1, v, 0xCC);      \
    v3 = _mm256_blend_ps(t3, v, 0x33);      \
    v  = _mm256_shuffle_ps(t4, t6, 0x4E);   \
    v4 = _mm256_blend_ps(t4, v, 0xCC);      \
    v5 = _mm256_blend_ps(t6, v, 0x33);      \
    v  = _mm256_shuffle_ps(t5, t7, 0x4E);   \
    v6 = _mm256_blend_ps(t5, v, 0xCC);      \
    v7 = _mm256_blend_ps(t7, v, 0x33);

#define _MM256_TRANSPOSE8_LEFT(v0, v1, v2, v3, v4, v5, v6, v7) \
  __m256 t0 = _mm256_unpacklo_ps(v0, v1);                      \
  __m256 t1 = _mm256_unpackhi_ps(v0, v1);                      \
  __m256 t2 = _mm256_unpacklo_ps(v2, v3);                      \
  __m256 t3 = _mm256_unpackhi_ps(v2, v3);                      \
  __m256 t4 = _mm256_unpacklo_ps(v4, v5);                      \
  __m256 t5 = _mm256_unpackhi_ps(v4, v5);                      \
  __m256 t6 = _mm256_unpacklo_ps(v6, v7);                      \
  __m256 t7 = _mm256_unpackhi_ps(v6, v7);                      \
  v0 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(1,0,1,0));          \
  v1 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(3,2,3,2));          \
  v2 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(1,0,1,0));          \
  v3 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(3,2,3,2));          \
  v4 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(1,0,1,0));          \
  v5 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(3,2,3,2));          \
  v6 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(1,0,1,0));          \
  v7 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(3,2,3,2));          \
  t0 = _mm256_permute2f128_ps(v0, v4, 0x20);                   \
  t1 = _mm256_permute2f128_ps(v1, v5, 0x20);                   \
  t2 = _mm256_permute2f128_ps(v2, v6, 0x20);                   \
  t3 = _mm256_permute2f128_ps(v3, v7, 0x20);                   \
  t4 = _mm256_permute2f128_ps(v0, v4, 0x31);                   \
  t5 = _mm256_permute2f128_ps(v1, v5, 0x31);                   \
  t6 = _mm256_permute2f128_ps(v2, v6, 0x31);                   \
  t7 = _mm256_permute2f128_ps(v3, v7, 0x31);

template <int left_c>
inline void PackC8_Left(float *dst, const float *src, size_t hw, size_t src_hw_stride) {
    auto src0 = src;
    auto src1 = src + src_hw_stride;
    auto src2 = src + src_hw_stride * 2;
    auto src3 = src + src_hw_stride * 3;
    auto src4 = src + src_hw_stride * 4;
    auto src5 = src + src_hw_stride * 5;
    auto src6 = src + src_hw_stride * 6;
    int cur_hw = 0;
#ifdef __AVX__
    __m256 v1 = _mm256_setzero_ps();
    __m256 v2 = _mm256_setzero_ps();
    __m256 v3 = _mm256_setzero_ps();
    __m256 v4 = _mm256_setzero_ps();
    __m256 v5 = _mm256_setzero_ps();
    __m256 v6 = _mm256_setzero_ps();
    __m256 v7 = _mm256_setzero_ps();

    for (; cur_hw + 7 < hw; cur_hw += 8) {
        auto dst_hw = dst + cur_hw * 8;
        __m256 v0 = _mm256_loadu_ps(src0 + cur_hw);
        if (left_c > 1) v1 = _mm256_loadu_ps(src1 + cur_hw);
        if (left_c > 2) v2 = _mm256_loadu_ps(src2 + cur_hw);
        if (left_c > 3) v3 = _mm256_loadu_ps(src3 + cur_hw);
        if (left_c > 4) v4 = _mm256_loadu_ps(src4 + cur_hw);
        if (left_c > 5) v5 = _mm256_loadu_ps(src5 + cur_hw);
        if (left_c > 6) v6 = _mm256_loadu_ps(src6 + cur_hw);
        _MM256_TRANSPOSE8_LEFT(v0, v1, v2, v3, v4, v5, v6, v7);
        _mm256_storeu_ps(dst_hw, t0);
        _mm256_storeu_ps(dst_hw + 8, t1);
        _mm256_storeu_ps(dst_hw + 16, t2);
        _mm256_storeu_ps(dst_hw + 24, t3);
        _mm256_storeu_ps(dst_hw + 32, t4);
        _mm256_storeu_ps(dst_hw + 40, t5);
        _mm256_storeu_ps(dst_hw + 48, t6);
        _mm256_storeu_ps(dst_hw + 56, t7);
    }
#endif
    for (; cur_hw < hw; cur_hw++) {
        dst[cur_hw * 8 + 0] = src0[cur_hw];
        if (left_c > 1) {
            dst[cur_hw * 8 + 1] = src1[cur_hw];
        } else {
            dst[cur_hw * 8 + 1] = 0;
        }
        if (left_c > 2) {
            dst[cur_hw * 8 + 2] = src2[cur_hw];
        } else {
            dst[cur_hw * 8 + 2] = 0;
        }
        if (left_c > 3) {
            dst[cur_hw * 8 + 3] = src3[cur_hw];
        } else {
            dst[cur_hw * 8 + 3] = 0;
        }
        if (left_c > 4) {
            dst[cur_hw * 8 + 4] = src4[cur_hw];
        } else {
            dst[cur_hw * 8 + 4] = 0;
        }
        if (left_c > 5) {
            dst[cur_hw * 8 + 5] = src5[cur_hw];
        } else {
            dst[cur_hw * 8 + 5] = 0;
        }
        if (left_c > 6) {
            dst[cur_hw * 8 + 6] = src6[cur_hw];
        } else {
            dst[cur_hw * 8 + 6] = 0;
        }
       dst[cur_hw * 8 + 7]  = 0;
    }
}
int PackC8(float *dst, const float *src, size_t hw, size_t src_hw_stride, size_t dst_hw_stride, size_t channel) {
    int c = 0;
    for (; c + 7 < channel; c += 8) {
        auto src0 = src + c * src_hw_stride;
        auto src1 = src0 + src_hw_stride;
        auto src2 = src0 + src_hw_stride * 2;
        auto src3 = src0 + src_hw_stride * 3;
        auto src4 = src0 + src_hw_stride * 4;
        auto src5 = src0 + src_hw_stride * 5;
        auto src6 = src0 + src_hw_stride * 6;
        auto src7 = src0 + src_hw_stride * 7;
        auto dst_c = dst + c * dst_hw_stride;
        int cur_hw = 0;
#ifdef __AVX__
        for (; cur_hw + 7 < hw; cur_hw += 8) {
            auto dst_hw = dst_c + cur_hw * 8;
            __m256 v0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(src0 + cur_hw)),     _mm_loadu_ps(src4 + cur_hw), 1);
            __m256 v1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(src1 + cur_hw)),     _mm_loadu_ps(src5 + cur_hw), 1);
            __m256 v2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(src2 + cur_hw)),     _mm_loadu_ps(src6 + cur_hw), 1);
            __m256 v3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(src3 + cur_hw)),     _mm_loadu_ps(src7 + cur_hw), 1);
            __m256 v4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(src0 + cur_hw + 4)), _mm_loadu_ps(src4 + cur_hw + 4), 1);
            __m256 v5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(src1 + cur_hw + 4)), _mm_loadu_ps(src5 + cur_hw + 4), 1);
            __m256 v6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(src2 + cur_hw + 4)), _mm_loadu_ps(src6 + cur_hw + 4), 1);
            __m256 v7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(src3 + cur_hw + 4)), _mm_loadu_ps(src7 + cur_hw + 4), 1);
            _MM256_TRANSPOSE8(v0, v1, v2, v3, v4, v5, v6, v7);
            _mm256_storeu_ps(dst_hw, v0);
            _mm256_storeu_ps(dst_hw + 8, v1);
            _mm256_storeu_ps(dst_hw + 16, v2);
            _mm256_storeu_ps(dst_hw + 24, v3);
            _mm256_storeu_ps(dst_hw + 32, v4);
            _mm256_storeu_ps(dst_hw + 40, v5);
            _mm256_storeu_ps(dst_hw + 48, v6);
            _mm256_storeu_ps(dst_hw + 56, v7);
        }
#endif
        for (; cur_hw < hw; cur_hw++) {
            dst_c[cur_hw * 8 + 0] = src0[cur_hw];
            dst_c[cur_hw * 8 + 1] = src1[cur_hw];
            dst_c[cur_hw * 8 + 2] = src2[cur_hw];
            dst_c[cur_hw * 8 + 3] = src3[cur_hw];
            dst_c[cur_hw * 8 + 4] = src4[cur_hw];
            dst_c[cur_hw * 8 + 5] = src5[cur_hw];
            dst_c[cur_hw * 8 + 6] = src6[cur_hw];
            dst_c[cur_hw * 8 + 7] = src7[cur_hw];
        }
    }
    int left_c = channel - c;
    if (left_c == 7) {
        PackC8_Left<7>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, src_hw_stride);
    } else if (left_c == 6) {
        PackC8_Left<6>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, src_hw_stride);
    } else if (left_c == 5) {
        PackC8_Left<5>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, src_hw_stride);
    } else if (left_c == 4) {
        PackC8_Left<4>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, src_hw_stride);
    } else if (left_c == 3) {
        PackC8_Left<3>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, src_hw_stride);
    } else if (left_c == 2) {
        PackC8_Left<2>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, src_hw_stride);
    } else if (left_c == 1) {
        PackC8_Left<1>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, src_hw_stride);
    }
    return 0;
}

template <int left_c>
inline void UnpackC4_Left(float *dst, const float *src, size_t hw, size_t dst_hw_stride) {
    auto dst0 = dst;
    auto dst1 = dst + dst_hw_stride;
    auto dst2 = dst + dst_hw_stride * 2;
    int cur_hw = 0;
    for (; cur_hw + 3 < hw; cur_hw += 4) {
        auto src_hw = src + cur_hw * 4;
        __m128 v0 = _mm_load_ps(src_hw);
        __m128 v1 = _mm_load_ps(src_hw + 4);
        __m128 v2 = _mm_load_ps(src_hw + 8);
        __m128 v3 = _mm_load_ps(src_hw + 12);
        _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
        _mm_storeu_ps(dst0 + cur_hw, v0);
        if (left_c > 1) _mm_storeu_ps(dst1 + cur_hw, v1);
        if (left_c > 2) _mm_storeu_ps(dst2 + cur_hw, v2);
    }
    for (; cur_hw < hw; cur_hw++) {
        dst0[cur_hw] = src[cur_hw * 4 + 0];
        if (left_c > 1) dst1[cur_hw] = src[cur_hw * 4 + 1];
        if (left_c > 2) dst2[cur_hw] = src[cur_hw * 4 + 2];
    }
}

int UnpackC4(float *dst, const float *src, size_t hw, size_t src_hw_stride, size_t dst_hw_stride, size_t channel) {
    int c = 0;
    for (; c + 3 < channel; c += 4) {
        auto src_c = src + c * src_hw_stride;
        auto dst0 = dst + c * dst_hw_stride;
        auto dst1 = dst0 + dst_hw_stride;
        auto dst2 = dst0 + dst_hw_stride * 2;
        auto dst3 = dst0 + dst_hw_stride * 3;
        int cur_hw = 0;
        for (; cur_hw + 3 < hw; cur_hw += 4) {
            auto src_hw = src_c + cur_hw * 4;
            __m128 v0 = _mm_load_ps(src_hw);
            __m128 v1 = _mm_load_ps(src_hw + 4);
            __m128 v2 = _mm_load_ps(src_hw + 8);
            __m128 v3 = _mm_load_ps(src_hw + 12);
            _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
            _mm_storeu_ps(dst0 + cur_hw, v0);
            _mm_storeu_ps(dst1 + cur_hw, v1);
            _mm_storeu_ps(dst2 + cur_hw, v2);
            _mm_storeu_ps(dst3 + cur_hw, v3);
        }
        for (; cur_hw < hw; cur_hw++) {
            dst0[cur_hw] = src_c[cur_hw * 4 + 0];
            dst1[cur_hw] = src_c[cur_hw * 4 + 1];
            dst2[cur_hw] = src_c[cur_hw * 4 + 2];
            dst3[cur_hw] = src_c[cur_hw * 4 + 3];
        }
    }
    int left_c = channel - c;
    if (left_c == 3) {
        UnpackC4_Left<3>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, dst_hw_stride);
    } else if (left_c == 2) {
        UnpackC4_Left<2>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, dst_hw_stride);
    } else if (left_c == 1) {
        UnpackC4_Left<1>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, dst_hw_stride);
    }
    return 0;
}

template <int left_c>
inline void UnpackC8_Left(float *dst, const float *src, size_t hw, size_t dst_hw_stride) {
    auto dst0 = dst;
    auto dst1 = dst + dst_hw_stride;
    auto dst2 = dst + dst_hw_stride * 2;
    auto dst3 = dst + dst_hw_stride * 3;
    auto dst4 = dst + dst_hw_stride * 4;
    auto dst5 = dst + dst_hw_stride * 5;
    auto dst6 = dst + dst_hw_stride * 6;
    int cur_hw = 0;
#ifdef __AVX__
    for (; cur_hw + 7 < hw; cur_hw += 8) {
        auto src_hw = src + cur_hw * 8;
        __m256 v0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw)),      _mm_load_ps(src_hw + 32), 1);
        __m256 v1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 8)),  _mm_load_ps(src_hw + 40), 1);
        __m256 v2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 16)), _mm_load_ps(src_hw + 48), 1);
        __m256 v3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 24)), _mm_load_ps(src_hw + 56), 1);
        __m256 v4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 4)),  _mm_load_ps(src_hw + 36), 1);
        __m256 v5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 12)), _mm_load_ps(src_hw + 44), 1);
        __m256 v6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 20)), _mm_load_ps(src_hw + 52), 1);
        __m256 v7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 28)), _mm_load_ps(src_hw + 60), 1);
        _MM256_TRANSPOSE8(v0, v1, v2, v3, v4, v5, v6, v7);
        _mm256_storeu_ps(dst0 + cur_hw, v0);
        if(left_c > 1) _mm256_storeu_ps(dst1 + cur_hw, v1);
        if(left_c > 2) _mm256_storeu_ps(dst2 + cur_hw, v2);
        if(left_c > 3) _mm256_storeu_ps(dst3 + cur_hw, v3);
        if(left_c > 4) _mm256_storeu_ps(dst4 + cur_hw, v4);
        if(left_c > 5) _mm256_storeu_ps(dst5 + cur_hw, v5);
        if(left_c > 6) _mm256_storeu_ps(dst6 + cur_hw, v6);
    }
#endif
    for (; cur_hw < hw; cur_hw++) {
        dst0[cur_hw] = src[cur_hw * 8 + 0];
        if (left_c > 1) dst1[cur_hw] = src[cur_hw * 8 + 1];
        if (left_c > 2) dst2[cur_hw] = src[cur_hw * 8 + 2];
        if (left_c > 3) dst3[cur_hw] = src[cur_hw * 8 + 3];
        if (left_c > 4) dst4[cur_hw] = src[cur_hw * 8 + 4];
        if (left_c > 5) dst5[cur_hw] = src[cur_hw * 8 + 5];
        if (left_c > 6) dst6[cur_hw] = src[cur_hw * 8 + 6];
    }
}

int UnpackC8(float *dst, const float *src, size_t hw, size_t src_hw_stride, size_t dst_hw_stride, size_t channel) {
    int c = 0;
    for (; c + 7 < channel; c += 8) {
        auto src_c = src + c * src_hw_stride;
        auto dst0 = dst + c * dst_hw_stride;
        auto dst1 = dst0 + dst_hw_stride;
        auto dst2 = dst0 + dst_hw_stride * 2;
        auto dst3 = dst0 + dst_hw_stride * 3;
        auto dst4 = dst0 + dst_hw_stride * 4;
        auto dst5 = dst0 + dst_hw_stride * 5;
        auto dst6 = dst0 + dst_hw_stride * 6;
        auto dst7 = dst0 + dst_hw_stride * 7;
        int cur_hw = 0;
#ifdef __AVX__
        for (; cur_hw + 7 < hw; cur_hw += 8) {
            auto src_hw = src_c + cur_hw * 8;
            __m256 v0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw)),      _mm_load_ps(src_hw + 32), 1);
            __m256 v1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 8)),  _mm_load_ps(src_hw + 40), 1);
            __m256 v2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 16)), _mm_load_ps(src_hw + 48), 1);
            __m256 v3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 24)), _mm_load_ps(src_hw + 56), 1);
            __m256 v4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 4)),  _mm_load_ps(src_hw + 36), 1);
            __m256 v5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 12)), _mm_load_ps(src_hw + 44), 1);
            __m256 v6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 20)), _mm_load_ps(src_hw + 52), 1);
            __m256 v7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(src_hw + 28)), _mm_load_ps(src_hw + 60), 1);
            _MM256_TRANSPOSE8(v0, v1, v2, v3, v4, v5, v6, v7);
            _mm256_storeu_ps(dst0 + cur_hw, v0);
            _mm256_storeu_ps(dst1 + cur_hw, v1);
            _mm256_storeu_ps(dst2 + cur_hw, v2);
            _mm256_storeu_ps(dst3 + cur_hw, v3);
            _mm256_storeu_ps(dst4 + cur_hw, v4);
            _mm256_storeu_ps(dst5 + cur_hw, v5);
            _mm256_storeu_ps(dst6 + cur_hw, v6);
            _mm256_storeu_ps(dst7 + cur_hw, v7);
        }
#endif
        for (; cur_hw < hw; cur_hw++) {
            dst0[cur_hw] = src_c[cur_hw * 8 + 0];
            dst1[cur_hw] = src_c[cur_hw * 8 + 1];
            dst2[cur_hw] = src_c[cur_hw * 8 + 2];
            dst3[cur_hw] = src_c[cur_hw * 8 + 3];
            dst4[cur_hw] = src_c[cur_hw * 8 + 4];
            dst5[cur_hw] = src_c[cur_hw * 8 + 5];
            dst6[cur_hw] = src_c[cur_hw * 8 + 6];
            dst7[cur_hw] = src_c[cur_hw * 8 + 7];
        }
    }
    int left_c = channel - c;
    if (left_c == 7) {
        UnpackC8_Left<7>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, dst_hw_stride);
    } else if (left_c == 6) {
        UnpackC8_Left<6>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, dst_hw_stride);
    } else if (left_c == 5) {
        UnpackC8_Left<5>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, dst_hw_stride);
    } else if (left_c == 4) {
        UnpackC8_Left<4>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, dst_hw_stride);
    } else if (left_c == 3) {
        UnpackC8_Left<3>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, dst_hw_stride);
    } else if (left_c == 2) {
        UnpackC8_Left<2>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, dst_hw_stride);
    } else if (left_c == 1) {
        UnpackC8_Left<1>(dst + c * dst_hw_stride, src + c * src_hw_stride, hw, dst_hw_stride);
    }
    return 0;
}

template<typename T>
int MatTranspose(T *dst, const T *src, size_t M, size_t N) {
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            dst[n * M + m] = src[m * N + n];
        }
    }

    return 0;
}
template int MatTranspose(float *dst, const float *src, size_t M, size_t N);

// from [o][i][h][w]
// to: [o/4][h][w][i/16][o4][i16]
int PackINT8Weight(int8_t *src, int8_t *dst, int input_channel, int output_channel, int height, int width) {
    const int oc_4        = (output_channel + 3) / 4;
    const int ic_calc     = input_channel < 4 ? input_channel : ROUND_UP(input_channel, 4);
    const int crs_round16 = ROUND_UP(ic_calc * height * width, 16);
    memset(dst, 0, oc_4 * 4 * crs_round16);
    for (int o = 0; o < output_channel; o++) {
        auto zo = o / 4, ro = o % 4;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int i = 0; i < input_channel; i++) {
                    // to: [o/4][h][w][i/16][o4][i16]
                    auto o_dst = dst + zo * 4 * crs_round16 + ro * 16;
                    auto ri    = ((h * width + w) * ic_calc + i) % 16;
                    auto zi    = ((h * width + w) * ic_calc + i) / 16;
                    o_dst[zi * 16 * 4 + ri] =
                        src[o * input_channel * height * width + i * height * width + h * width + w];
                }
            }
        }
    }
    return 0;
}

}  // namespace x86
}  // namespace TNN
