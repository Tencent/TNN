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

#include "tnn/core/macro.h"
#include "tnn/device/x86/x86_common.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/mat_converter_utils.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
namespace x86 {

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

#ifdef __SSE4_2__

static inline __m128i load_element_c4(const uint8_t* addr) {
    return _mm_loadl_epi64((__m128i*)addr);
}

static inline __m128i load_element_c3(const uint8_t* addr) {
    __m128i val;
    val = _mm_insert_epi32(val, *(int*)addr, 0);
    return _mm_insert_epi16(val, *(short*)(addr + 4), 2);
}

static inline __m128i load_element_c3_pack4(const uint8_t* addr) {
    __m128i val = load_element_c3(addr);
    return _mm_shuffle_epi8(val, _mm_setr_epi8(0, 1, 2, 6, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15));
}

static inline __m128i load_element_c2(const uint8_t* addr) {
    __m128i val;
    return _mm_insert_epi32(val, *(int*)addr, 0);
}

#endif

/*
color convert
*/

// float
//     r = 1.164 * (y - 16) + 1.596 * (v - 128);
//     g = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128);
//     b = 1.164 * (y - 16) + 2.018 * (u - 128);
// int 16
//     r = (74 * y - 1135 + 102 * vv ) >> 6
//     g = (74 * y - 1135 - 52 * vv - 25 * uu ) >> 6
//     b = (74 * y - 1135 + 129 * uu ) >> 6
template <bool is_nv12, bool has_alpha>
void YUVToBGR(const unsigned char* yuv, unsigned char* bgr, int h, int w) {
    const unsigned char* yptr  = yuv;
    const unsigned char* vuptr = yuv + w * h;
    const int channel          = has_alpha ? 4 : 3;

#ifdef __SSE4_2__
    __m128i _v1135 = _mm_set1_epi16(-1135);
    __m128i _v74   = _mm_set1_epi16(74);
    __m128i _v128  = _mm_set1_epi16(128);
    __m128i _v102  = _mm_set1_epi16(102);
    __m128i _v52   = _mm_set1_epi16(-52);
    __m128i _v25   = _mm_set1_epi16(-25);
    __m128i _v129  = _mm_set1_epi16(129);
    __m128i _v240  = _mm_set1_epi8(0xf0); // 240
    __m128i _aa    = _mm_set1_epi8(0xff); // 255

    const __m128i sh_vu = _mm_setr_epi8(0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15);
    const __m128i sh_a  = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
    const __m128i sh_b  = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
    const __m128i sh_c  = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
    const __m128i m0    = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    const __m128i m1    = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
#endif

    for (int y = 0; y < h; y += 2) {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* rgb0        = bgr;
        unsigned char* rgb1        = bgr + w * channel;

        int remain = w;
#ifdef __SSE4_2__
        int nn = w >> 4;
        remain = w - (nn << 4);
        for (; nn > 0; nn--) {
            __m128i _yy00_load = _mm_loadl_epi64((__m128i*)yptr0);
            __m128i _yy01_load = _mm_loadl_epi64((__m128i*)(yptr0 + 8));
            __m128i _yy10_load = _mm_loadl_epi64((__m128i*)yptr1);
            __m128i _yy11_load = _mm_loadl_epi64((__m128i*)(yptr1 + 8));
            __m128i _vu00_load = _mm_loadl_epi64((__m128i*)vuptr);
            __m128i _vu01_load = _mm_loadl_epi64((__m128i*)(vuptr + 8));
            _vu00_load         = _mm_min_epu8(_v240, _vu00_load);
            _vu01_load         = _mm_min_epu8(_v240, _vu01_load);

            __m128i _yy00 = _mm_add_epi16(_mm_mullo_epi16(_mm_cvtepu8_epi16(_yy00_load), _v74), _v1135);
            __m128i _yy01 = _mm_add_epi16(_mm_mullo_epi16(_mm_cvtepu8_epi16(_yy01_load), _v74), _v1135);
            __m128i _yy10 = _mm_add_epi16(_mm_mullo_epi16(_mm_cvtepu8_epi16(_yy10_load), _v74), _v1135);
            __m128i _yy11 = _mm_add_epi16(_mm_mullo_epi16(_mm_cvtepu8_epi16(_yy11_load), _v74), _v1135);

            __m128i _vu00 = _mm_sub_epi16(_mm_cvtepu8_epi16(_mm_shuffle_epi8(_vu00_load, sh_vu)), _v128);
            __m128i _vu01 = _mm_sub_epi16(_mm_cvtepu8_epi16(_mm_shuffle_epi8(_vu01_load, sh_vu)), _v128);

            // nv12 u,v,u,v,...
            __m128i _uu00 = is_nv12 ? _mm_unpacklo_epi16(_vu00, _vu00) : _mm_unpackhi_epi16(_vu00, _vu00);
            __m128i _vv00 = is_nv12 ? _mm_unpackhi_epi16(_vu00, _vu00) : _mm_unpacklo_epi16(_vu00, _vu00);
            __m128i _uu01 = is_nv12 ? _mm_unpacklo_epi16(_vu01, _vu01) : _mm_unpackhi_epi16(_vu01, _vu01);
            __m128i _vv01 = is_nv12 ? _mm_unpackhi_epi16(_vu01, _vu01) : _mm_unpacklo_epi16(_vu01, _vu01);

            __m128i _r00 = _mm_add_epi16(_yy00, _mm_mullo_epi16(_vv00, _v102));
            __m128i _g00 = _mm_add_epi16(_yy00, _mm_mullo_epi16(_vv00, _v52));
            _g00         = _mm_add_epi16(_g00, _mm_mullo_epi16(_uu00, _v25));
            __m128i _b00 = _mm_add_epi16(_yy00, _mm_mullo_epi16(_uu00, _v129));

            __m128i _r01 = _mm_add_epi16(_yy01, _mm_mullo_epi16(_vv01, _v102));
            __m128i _g01 = _mm_add_epi16(_yy01, _mm_mullo_epi16(_vv01, _v52));
            _g01         = _mm_add_epi16(_g01, _mm_mullo_epi16(_uu01, _v25));
            __m128i _b01 = _mm_add_epi16(_yy01, _mm_mullo_epi16(_uu01, _v129));

            __m128i _r10 = _mm_add_epi16(_yy10, _mm_mullo_epi16(_vv00, _v102));
            __m128i _g10 = _mm_add_epi16(_yy10, _mm_mullo_epi16(_vv00, _v52));
            _g10         = _mm_add_epi16(_g10, _mm_mullo_epi16(_uu00, _v25));
            __m128i _b10 = _mm_add_epi16(_yy10, _mm_mullo_epi16(_uu00, _v129));

            __m128i _r11 = _mm_add_epi16(_yy11, _mm_mullo_epi16(_vv01, _v102));
            __m128i _g11 = _mm_add_epi16(_yy11, _mm_mullo_epi16(_vv01, _v52));
            _g11         = _mm_add_epi16(_g11, _mm_mullo_epi16(_uu01, _v25));
            __m128i _b11 = _mm_add_epi16(_yy11, _mm_mullo_epi16(_uu01, _v129));

            __m128i _r00_srai = _mm_srai_epi16(_r00, 6);
            __m128i _g00_srai = _mm_srai_epi16(_g00, 6);
            __m128i _b00_srai = _mm_srai_epi16(_b00, 6);
            __m128i _r01_srai = _mm_srai_epi16(_r01, 6);
            __m128i _g01_srai = _mm_srai_epi16(_g01, 6);
            __m128i _b01_srai = _mm_srai_epi16(_b01, 6);
            __m128i _r10_srai = _mm_srai_epi16(_r10, 6);
            __m128i _g10_srai = _mm_srai_epi16(_g10, 6);
            __m128i _b10_srai = _mm_srai_epi16(_b10, 6);
            __m128i _r11_srai = _mm_srai_epi16(_r11, 6);
            __m128i _g11_srai = _mm_srai_epi16(_g11, 6);
            __m128i _b11_srai = _mm_srai_epi16(_b11, 6);

            __m128i rr0 = _mm_packus_epi16(_r00_srai, _r01_srai);
            __m128i gg0 = _mm_packus_epi16(_g00_srai, _g01_srai);
            __m128i bb0 = _mm_packus_epi16(_b00_srai, _b01_srai);
            __m128i rr1 = _mm_packus_epi16(_r10_srai, _r11_srai);
            __m128i gg1 = _mm_packus_epi16(_g10_srai, _g11_srai);
            __m128i bb1 = _mm_packus_epi16(_b10_srai, _b11_srai);

            if (!has_alpha) {
                // bbbb, gggg, rrrr to bgr,bgr,bgr,bgr
                __m128i a0 = _mm_shuffle_epi8(bb0, sh_a);
                __m128i b0 = _mm_shuffle_epi8(gg0, sh_b);
                __m128i c0 = _mm_shuffle_epi8(rr0, sh_c);
                __m128i a1 = _mm_shuffle_epi8(bb1, sh_a);
                __m128i b1 = _mm_shuffle_epi8(gg1, sh_b);
                __m128i c1 = _mm_shuffle_epi8(rr1, sh_c);

                __m128i v0 = _mm_blendv_epi8(_mm_blendv_epi8(a0, b0, m1), c0, m0);
                __m128i v1 = _mm_blendv_epi8(_mm_blendv_epi8(b0, c0, m1), a0, m0);
                __m128i v2 = _mm_blendv_epi8(_mm_blendv_epi8(c0, a0, m1), b0, m0);
                __m128i v3 = _mm_blendv_epi8(_mm_blendv_epi8(a1, b1, m1), c1, m0);
                __m128i v4 = _mm_blendv_epi8(_mm_blendv_epi8(b1, c1, m1), a1, m0);
                __m128i v5 = _mm_blendv_epi8(_mm_blendv_epi8(c1, a1, m1), b1, m0);

                _mm_storeu_si128((__m128i*)rgb0, v0);
                _mm_storeu_si128((__m128i*)(rgb0 + 16), v1);
                _mm_storeu_si128((__m128i*)(rgb0 + 32), v2);
                _mm_storeu_si128((__m128i*)rgb1, v3);
                _mm_storeu_si128((__m128i*)(rgb1 + 16), v4);
                _mm_storeu_si128((__m128i*)(rgb1 + 32), v5);
            } else {
                // bbbb, gggg, rrrr, aaaa to bgra,bgra,bgra,bgra
                __m128i _bg, _ra, _res0, _res1, _res2, _res3;

                _bg   = _mm_unpacklo_epi8(bb0, gg0);
                _ra   = _mm_unpacklo_epi8(rr0, _aa);
                _res0 = _mm_unpacklo_epi16(_bg, _ra);
                _res1 = _mm_unpackhi_epi16(_bg, _ra);
                _bg   = _mm_unpackhi_epi8(bb0, gg0);
                _ra   = _mm_unpackhi_epi8(rr0, _aa);
                _res2 = _mm_unpacklo_epi16(_bg, _ra);
                _res3 = _mm_unpackhi_epi16(_bg, _ra);

                _mm_storeu_si128((__m128i*)rgb0, _res0);
                _mm_storeu_si128((__m128i*)(rgb0 + 16), _res1);
                _mm_storeu_si128((__m128i*)(rgb0 + 32), _res2);
                _mm_storeu_si128((__m128i*)(rgb0 + 48), _res3);

                _bg   = _mm_unpacklo_epi8(bb1, gg1);
                _ra   = _mm_unpacklo_epi8(rr1, _aa);
                _res0 = _mm_unpacklo_epi16(_bg, _ra);
                _res1 = _mm_unpackhi_epi16(_bg, _ra);
                _bg   = _mm_unpackhi_epi8(bb1, gg1);
                _ra   = _mm_unpackhi_epi8(rr1, _aa);
                _res2 = _mm_unpacklo_epi16(_bg, _ra);
                _res3 = _mm_unpackhi_epi16(_bg, _ra);

                _mm_storeu_si128((__m128i*)rgb1, _res0);
                _mm_storeu_si128((__m128i*)(rgb1 + 16), _res1);
                _mm_storeu_si128((__m128i*)(rgb1 + 32), _res2);
                _mm_storeu_si128((__m128i*)(rgb1 + 48), _res3);
            }

            yptr0 += 16;
            yptr1 += 16;
            vuptr += 16;
            rgb0 += 16 * channel;
            rgb1 += 16 * channel;
        }
#endif

        NaiveYUVToBGROrBGRALoop(yptr0, yptr1, vuptr, rgb0, rgb1, remain, is_nv12, channel);
        yptr += 2 * w;
        vuptr += remain;
        bgr += 2 * channel * w;
    }
}

void NV12ToBGR(const unsigned char* nv12, unsigned char* bgr, int h, int w) {
    return YUVToBGR<true, false>(nv12, bgr, h, w);
}
void NV21ToBGR(const unsigned char* nv21, unsigned char* bgr, int h, int w) {
    return YUVToBGR<false, false>(nv21, bgr, h, w);
}
void NV12ToBGRA(const unsigned char* nv12, unsigned char* bgra, int h, int w) {
    return YUVToBGR<true, true>(nv12, bgra, h, w);
}
void NV21ToBGRA(const unsigned char* nv21, unsigned char* bgra, int h, int w) {
    return YUVToBGR<false, true>(nv21, bgra, h, w);
}

template <int channel, bool bgr_order>
void ColorToGray(const unsigned char* bgr, unsigned char* gray, int h, int w) {
    int offset = 0;
    int plane  = h * w;

#ifdef __SSE4_2__
    const unsigned char* Sp = bgr;
    unsigned char* Dp       = gray;
    __m128 _coeff_b         = _mm_set1_ps(0.114);
    __m128 _coeff_g         = _mm_set1_ps(0.587);
    __m128 _coeff_r         = _mm_set1_ps(0.299);
    __m128i _vzero          = _mm_set1_epi16(0);
    __m128i _maski16_to_i8  = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    __m128i _maski32_to_i16 = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    __m128i _maskld3_0      = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
    __m128i _maskld3_1      = _mm_setr_epi8(2, 5, 0, 3, 6, 1, 4, 7, 0, 0, 0, 0, 0, 0, 0, 0);
    for (; offset<plane>> 3 << 3; offset += 8) {
        __m128i b_h, g_h, r_h;
        if (channel == 3) {
            __m128i _bgra0_load = _mm_loadu_si128((__m128i*)Sp);              // b0g0r0 b1g1r1 b2g2r2 b3g3r3 b4g4r4 b5
            __m128i _bgra1_load = _mm_loadl_epi64((__m128i*)(Sp + 16));       // g5r5 b6g6r6 b7g7r7
            __m128i tmp0        = _mm_shuffle_epi8(_bgra0_load, _maskld3_0);  // b0b1b2b3b4b5 g0g1g2g3g4 r0r1r2r3r4
            __m128i tmp1        = _mm_shuffle_epi8(_bgra1_load, _maskld3_1);  // b6b7 g5g6g7 r5r6r7
            _bgra0_load         = _mm_srli_si128(_mm_slli_si128(tmp0, 10), 10);  // b0b1b2b3b4b5 0
            _bgra1_load         = _mm_slli_si128(tmp1, 6);                       // 0 0 0 0 0 0  b6b7
            __m128i b_b         = _mm_or_si128(_bgra0_load, _bgra1_load);
            _bgra0_load         = _mm_srli_si128(_mm_slli_si128(tmp0, 5), 11);  // g0g1g2g3g4 0
            _bgra1_load         = _mm_slli_si128(_mm_srli_si128(tmp1, 2), 5);   // 0 0 0 0 0  g5g6g7
            __m128i g_b         = _mm_or_si128(_bgra0_load, _bgra1_load);
            _bgra0_load         = _mm_srli_si128(tmp0, 11);                    // r0r1r2r3r4 0
            _bgra1_load         = _mm_slli_si128(_mm_srli_si128(tmp1, 5), 5);  // 0 0 0 0 0  r5r6r7
            __m128i r_b         = _mm_or_si128(_bgra0_load, _bgra1_load);

            b_h = _mm_cvtepu8_epi16(bgr_order ? b_b : r_b);
            g_h = _mm_cvtepu8_epi16(g_b);
            r_h = _mm_cvtepu8_epi16(bgr_order ? r_b : b_b);
        } else {
            __m128i _bgra0_load = _mm_loadu_si128((__m128i*)Sp);                // a0 - a15
            __m128i _bgra1_load = _mm_loadu_si128((__m128i*)(Sp + 16));         // b0 - b15
            __m128i tmp0        = _mm_unpacklo_epi8(_bgra0_load, _bgra1_load);  // a0,b0, - a7,b7
            __m128i tmp1        = _mm_unpackhi_epi8(_bgra0_load, _bgra1_load);  // a8,b8, - a15,b15
            _bgra0_load         = _mm_unpacklo_epi8(tmp0, tmp1);                // a0,a8,b0,b8, - a3,a11,b3,b11
            _bgra1_load         = _mm_unpackhi_epi8(tmp0, tmp1);                // a4,a12,b4,b12, - a7,a15,b7,b15
            tmp0                = _mm_unpacklo_epi8(_bgra0_load, _bgra1_load);  // a0 - b12, a1 - b13
            tmp1                = _mm_unpackhi_epi8(_bgra0_load, _bgra1_load);  // a2 - b14, a3 - b15

            b_h = _mm_cvtepu8_epi16(bgr_order ? tmp0 : tmp1);
            g_h = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(tmp0, tmp0));
            r_h = _mm_cvtepu8_epi16(bgr_order ? tmp1 : tmp0);
        }

        __m128 b_val, g_val, r_val, acc;
        b_val        = _mm_cvtepi32_ps(_mm_unpacklo_epi16(b_h, _vzero));
        g_val        = _mm_cvtepi32_ps(_mm_unpacklo_epi16(g_h, _vzero));
        r_val        = _mm_cvtepi32_ps(_mm_unpacklo_epi16(r_h, _vzero));
        acc          = _mm_mul_ps(b_val, _coeff_b);
        acc          = _mm_add_ps(acc, _mm_mul_ps(g_val, _coeff_g));
        acc          = _mm_add_ps(acc, _mm_mul_ps(r_val, _coeff_r));
        __m128i tmp0 = _mm_shuffle_epi8(_mm_cvtps_epi32(acc), _maski32_to_i16);

        b_val        = _mm_cvtepi32_ps(_mm_unpackhi_epi16(b_h, _vzero));
        g_val        = _mm_cvtepi32_ps(_mm_unpackhi_epi16(g_h, _vzero));
        r_val        = _mm_cvtepi32_ps(_mm_unpackhi_epi16(r_h, _vzero));
        acc          = _mm_mul_ps(b_val, _coeff_b);
        acc          = _mm_add_ps(acc, _mm_mul_ps(g_val, _coeff_g));
        acc          = _mm_add_ps(acc, _mm_mul_ps(r_val, _coeff_r));
        __m128i tmp1 = _mm_shuffle_epi8(_mm_cvtps_epi32(acc), _maski32_to_i16);

        _mm_storel_epi64((__m128i*)Dp, _mm_shuffle_epi8(_mm_unpacklo_epi64(tmp0, tmp1), _maski16_to_i8));

        Sp += 8 * channel;
        Dp += 8;
    }
    if (plane % 8) {
        offset -= 8;
    }
#endif

    for (; offset < plane; ++offset) {
        unsigned b       = bgr[offset * channel + (bgr_order ? 0 : 2)];
        unsigned g       = bgr[offset * channel + 1];
        unsigned r       = bgr[offset * channel + (bgr_order ? 2 : 0)];
        float gray_color = 0.114 * b + 0.587 * g + 0.299 * r;
        gray[offset]     = gray_color;
    }
}

void BGRToGray(const unsigned char* bgr, unsigned char* gray, int height, int width) {
    ColorToGray<3, true>(bgr, gray, height, width);
}
void BGRAToGray(const unsigned char* bgra, unsigned char* gray, int height, int width) {
    ColorToGray<4, true>(bgra, gray, height, width);
}
void RGBToGray(const unsigned char* rgb, unsigned char* gray, int height, int width) {
    ColorToGray<3, false>(rgb, gray, height, width);
}
void RGBAToGray(const unsigned char* rgba, unsigned char* gray, int height, int width) {
    ColorToGray<4, false>(rgba, gray, height, width);
}

/*
resize
*/

template <int c>
static void ResizeGetAdjacentRows(int sy, int prev_sy, short** rows0, short** rows1, int* xofs, const uint8_t* src,
                                  int src_stride, int w, const short* ialphap) {
    if (sy == prev_sy) {
        // reuse all rows
    } else if (sy == prev_sy + 1) {
        // hresize one row
        short* rows0_old  = *rows0;
        *rows0            = *rows1;
        *rows1            = rows0_old;
        const uint8_t* S1 = src + src_stride * (sy + 1);

        short* rows1p = *rows1;
        for (int dx = 0; dx < w; dx++) {
            int sx   = xofs[dx];
            short a0 = ialphap[0];
            short a1 = ialphap[1];

            const uint8_t* S1p = S1 + sx;

#ifndef __SSE4_2__
            for (int dc = 0; dc < c; ++dc) {
                rows1p[dc] = (S1p[dc] * a0 + S1p[dc + c] * a1) >> 4;
            }
#else
            __m128i _maski32_to_i16 = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
            if (c == 2) {
                __m128i _a0 = _mm_set1_epi16(a0);
                __m128i _a1 = _mm_set1_epi16(a1);
                __m128i _S1 = _mm_cvtepu8_epi16(load_element_c2(S1p));
                __m128i _Sh = _mm_srli_si128(_S1, 4);

                __m128i _res = _mm_madd_epi16(_mm_unpacklo_epi16(_a0, _a1), _mm_unpacklo_epi16(_S1, _Sh));
                _res         = _mm_shuffle_epi8(_mm_srai_epi32(_res, 4), _maski32_to_i16);
                _mm_storel_epi64((__m128i*)rows1p, _res);
            } else if (c == 3) {
                __m128i _a0 = _mm_set1_epi16(a0);
                __m128i _a1 = _mm_set1_epi16(a1);
                __m128i _S1 = _mm_cvtepu8_epi16(load_element_c3(S1p));
                __m128i _Sh = _mm_srli_si128(_S1, 6);

                __m128i _res = _mm_madd_epi16(_mm_unpacklo_epi16(_a0, _a1), _mm_unpacklo_epi16(_S1, _Sh));
                _res         = _mm_shuffle_epi8(_mm_srai_epi32(_res, 4), _maski32_to_i16);
                _mm_storel_epi64((__m128i*)rows1p, _res);
            } else if (c == 4) {
                __m128i _a0 = _mm_set1_epi16(a0);
                __m128i _a1 = _mm_set1_epi16(a1);
                __m128i _S1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)S1p));
                __m128i _Sh = _mm_unpackhi_epi64(_S1, _S1);

                __m128i _res = _mm_madd_epi16(_mm_unpacklo_epi16(_a0, _a1), _mm_unpacklo_epi16(_S1, _Sh));
                _res         = _mm_shuffle_epi8(_mm_srai_epi32(_res, 4), _maski32_to_i16);
                _mm_storel_epi64((__m128i*)rows1p, _res);
            } else {
                for (int dc = 0; dc < c; ++dc) {
                    rows1p[dc] = (S1p[dc] * a0 + S1p[dc + c] * a1) >> 4;
                }
            }
#endif

            ialphap += 2;
            rows1p += c;
        }
    } else {
        // hresize two rows
        const uint8_t* S0 = src + src_stride * (sy);
        const uint8_t* S1 = src + src_stride * (sy + 1);

        short* rows0p = *rows0;
        short* rows1p = *rows1;
        for (int dx = 0; dx < w; dx++) {
            int sx   = xofs[dx];
            short a0 = ialphap[0];
            short a1 = ialphap[1];

            const uint8_t* S0p = S0 + sx;
            const uint8_t* S1p = S1 + sx;

#ifndef __SSE4_2__
            for (int dc = 0; dc < c; ++dc) {
                rows0p[dc] = (S0p[dc] * a0 + S0p[dc + c] * a1) >> 4;
                rows1p[dc] = (S1p[dc] * a0 + S1p[dc + c] * a1) >> 4;
            }
#else
            __m128i _maski32_to_i16 = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
            if (c == 2) {
                __m128i _a0  = _mm_set1_epi16(a0);
                __m128i _a1  = _mm_set1_epi16(a1);
                __m128i _S0  = _mm_cvtepu8_epi16(load_element_c2(S0p));
                __m128i _S1  = _mm_cvtepu8_epi16(load_element_c2(S1p));
                __m128i _Sh0 = _mm_srli_si128(_S0, 4);
                __m128i _Sh1 = _mm_srli_si128(_S1, 4);

                __m128i _res0 = _mm_madd_epi16(_mm_unpacklo_epi16(_a0, _a1), _mm_unpacklo_epi16(_S0, _Sh0));
                __m128i _res1 = _mm_madd_epi16(_mm_unpacklo_epi16(_a0, _a1), _mm_unpacklo_epi16(_S1, _Sh1));
                _res0         = _mm_shuffle_epi8(_mm_srai_epi32(_res0, 4), _maski32_to_i16);
                _res1         = _mm_shuffle_epi8(_mm_srai_epi32(_res1, 4), _maski32_to_i16);
                _mm_storel_epi64((__m128i*)rows0p, _res0);
                _mm_storel_epi64((__m128i*)rows1p, _res1);
            } else if (c == 3) {
                __m128i _a0  = _mm_set1_epi16(a0);
                __m128i _a1  = _mm_set1_epi16(a1);
                __m128i _S0  = _mm_cvtepu8_epi16(load_element_c3(S0p));
                __m128i _S1  = _mm_cvtepu8_epi16(load_element_c3(S1p));
                __m128i _Sh0 = _mm_srli_si128(_S0, 6);
                __m128i _Sh1 = _mm_srli_si128(_S1, 6);

                __m128i _res0 = _mm_madd_epi16(_mm_unpacklo_epi16(_a0, _a1), _mm_unpacklo_epi16(_S0, _Sh0));
                __m128i _res1 = _mm_madd_epi16(_mm_unpacklo_epi16(_a0, _a1), _mm_unpacklo_epi16(_S1, _Sh1));
                _res0         = _mm_shuffle_epi8(_mm_srai_epi32(_res0, 4), _maski32_to_i16);
                _res1         = _mm_shuffle_epi8(_mm_srai_epi32(_res1, 4), _maski32_to_i16);
                _mm_storel_epi64((__m128i*)rows0p, _res0);
                _mm_storel_epi64((__m128i*)rows1p, _res1);
            } else if (c == 4) {
                __m128i _a0  = _mm_set1_epi16(a0);
                __m128i _a1  = _mm_set1_epi16(a1);
                __m128i _S0  = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)S0p));
                __m128i _S1  = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)S1p));
                __m128i _Sh0 = _mm_unpackhi_epi64(_S0, _S0);
                __m128i _Sh1 = _mm_unpackhi_epi64(_S1, _S1);

                __m128i _res0 = _mm_madd_epi16(_mm_unpacklo_epi16(_a0, _a1), _mm_unpacklo_epi16(_S0, _Sh0));
                __m128i _res1 = _mm_madd_epi16(_mm_unpacklo_epi16(_a0, _a1), _mm_unpacklo_epi16(_S1, _Sh1));
                _res0         = _mm_shuffle_epi8(_mm_srai_epi32(_res0, 4), _maski32_to_i16);
                _res1         = _mm_shuffle_epi8(_mm_srai_epi32(_res1, 4), _maski32_to_i16);
                _mm_storel_epi64((__m128i*)rows0p, _res0);
                _mm_storel_epi64((__m128i*)rows1p, _res1);
            } else {
                for (int dc = 0; dc < c; ++dc) {
                    rows0p[dc] = (S0p[dc] * a0 + S0p[dc + c] * a1) >> 4;
                    rows1p[dc] = (S1p[dc] * a0 + S1p[dc + c] * a1) >> 4;
                }
            }
#endif

            ialphap += 2;
            rows0p += c;
            rows1p += c;
        }
    }
}

static void ResizeCalculateOneRow(short* rows0p, short* rows1p, const short b0, const short b1, const int w,
                                  const int c, uint8_t* Dp) {
#ifndef __SSE4_2__
    int remain = w * c;
#else
    int nn = (w * c) >> 4;
    int remain = (w * c) - (nn << 4);
    __m128i _b0 = _mm_set1_epi16(b0);
    __m128i _b1 = _mm_set1_epi16(b1);
    __m128i _v2 = _mm_set1_epi16(2);
    for (; nn > 0; nn--) {
        __m128i _rows0p_sr8 = _mm_loadu_si128((__m128i*)rows0p);
        __m128i _rows1p_sr8 = _mm_loadu_si128((__m128i*)rows1p);
        __m128i _rows0p_1_sr8 = _mm_loadu_si128((__m128i*)(rows0p + 8));
        __m128i _rows1p_1_sr8 = _mm_loadu_si128((__m128i*)(rows1p + 8));

        __m128i _rows0p_sr8_hi = _mm_mulhi_epi16(_rows0p_sr8, _b0);
        __m128i _rows1p_sr8_hi = _mm_mulhi_epi16(_rows1p_sr8, _b1);
        __m128i _rows0p_1_sr8_hi = _mm_mulhi_epi16(_rows0p_1_sr8, _b0);
        __m128i _rows1p_1_sr8_hi = _mm_mulhi_epi16(_rows1p_1_sr8, _b1);

        __m128i _acc = _mm_adds_epi16(_rows0p_sr8_hi, _rows1p_sr8_hi);
        __m128i _acc_1 = _mm_adds_epi16(_rows0p_1_sr8_hi, _rows1p_1_sr8_hi);
        _acc = _mm_srai_epi16(_mm_adds_epi16(_acc, _v2), 2);
        _acc_1 = _mm_srai_epi16(_mm_adds_epi16(_acc_1, _v2), 2);

        _mm_storeu_si128((__m128i*)Dp, _mm_packus_epi16(_acc, _acc_1));

        Dp += 16;
        rows0p += 16;
        rows1p += 16;
    }
#endif
    for (; remain; --remain) {
        *Dp++ =
            (uint8_t)(((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
    }
}

struct ResizeBilinearKernelParm {
    ResizeBilinearKernelParm(int* _xofs, int* _yofs, short* _ialpha, short* _ibeta, const uint8_t* _src, uint8_t* _dst,
                             int _src_plane, int _src_stride, int _schannel) {
        xofs       = _xofs;
        yofs       = _yofs;
        ialpha     = _ialpha;
        ibeta      = _ibeta;
        src        = _src;
        dst        = _dst;
        src_plane  = _src_plane;
        src_stride = _src_stride;
        schannel   = _schannel;
    };

    int* xofs;
    int* yofs;
    short* ialpha;
    short* ibeta;
    const uint8_t* src;
    uint8_t* dst;
    int src_plane;
    int src_stride;
    int schannel;
};

template <int channel>
void ResizeBilinearOneRow(ResizeBilinearKernelParm& param, int thread_id, short** rows0_t, short** rows1_t,
                          int* prev_sy, int b, int w, int h, int stride, int dy) {
    int sy = param.yofs[dy];
    ResizeGetAdjacentRows<channel>(sy, prev_sy[thread_id], &rows0_t[thread_id], &rows1_t[thread_id], param.xofs,
                                   param.src + b * param.src_plane, param.src_stride, w, param.ialpha);
    prev_sy[thread_id] = sy;

    // vresize
    short b0 = param.ibeta[dy * 2];
    short b1 = param.ibeta[dy * 2 + 1];

    uint8_t* Dp = param.dst + stride * (b * h + dy);

    ResizeCalculateOneRow(rows0_t[thread_id], rows1_t[thread_id], b0, b1, w, channel, Dp);
}

#define ResizeBilinearPreparation(channel)                                                                             \
    int schannel = channel;                                                                                            \
    int* buf     = nullptr;                                                                                            \
    GetResizeBuf(src_w, src_h, w, h, schannel, &buf);                                                                  \
    int* xofs     = buf;                                                                                               \
    int* yofs     = buf + w;                                                                                           \
    short* ialpha = (short*)(buf + w + h);                                                                             \
    short* ibeta  = (short*)(buf + w + h + w);                                                                         \
    int src_plane = src_h * src_stride;

void ResizeBilinearC1Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride, uint8_t* dst, int w,
                          int h, int stride) {
    ResizeBilinearPreparation(1);

    ResizeBilinearKernelParm param(xofs, yofs, ialpha, ibeta, src, dst, src_plane, src_stride, schannel);

    // loop body
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    short* rows0        = new short[w * max_num_threads];
    short* rows1        = new short[w * max_num_threads];
    short** rows0_t     = new short*[max_num_threads];
    short** rows1_t     = new short*[max_num_threads];
    int* prev_sy        = new int[max_num_threads];

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < max_num_threads; ++t) {
            prev_sy[t] = -2;
            rows0_t[t] = rows0 + t * w;
            rows1_t[t] = rows1 + t * w;
        }

        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            int thread_id = OMP_TID_;
            ResizeBilinearOneRow<1>(param, thread_id, rows0_t, rows1_t, prev_sy, b, w, h, stride, dy);
        }
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
    delete[] rows0_t;
    delete[] rows1_t;
    delete[] prev_sy;
}

void ResizeBilinearC2Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride, uint8_t* dst, int w,
                          int h, int stride) {
    ResizeBilinearPreparation(2);

    ResizeBilinearKernelParm param(xofs, yofs, ialpha, ibeta, src, dst, src_plane, src_stride, schannel);

    // loop body
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    short* rows0        = new short[(w * 2 + 2) * max_num_threads];
    short* rows1        = new short[(w * 2 + 2) * max_num_threads];
    short** rows0_t     = new short*[max_num_threads];
    short** rows1_t     = new short*[max_num_threads];
    int* prev_sy        = new int[max_num_threads];

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < max_num_threads; ++t) {
            prev_sy[t] = -2;
            rows0_t[t] = rows0 + t * (w * 2 + 2);
            rows1_t[t] = rows1 + t * (w * 2 + 2);
        }

        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            int thread_id = OMP_TID_;
            ResizeBilinearOneRow<2>(param, thread_id, rows0_t, rows1_t, prev_sy, b, w, h, stride, dy);
        }
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
    delete[] rows0_t;
    delete[] rows1_t;
    delete[] prev_sy;
}

void ResizeBilinearC3Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride, uint8_t* dst, int w,
                          int h, int stride) {
    ResizeBilinearPreparation(3);

    ResizeBilinearKernelParm param(xofs, yofs, ialpha, ibeta, src, dst, src_plane, src_stride, schannel);

    // loop body
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    short* rows0        = new short[(w * 3 + 1) * max_num_threads];
    short* rows1        = new short[(w * 3 + 1) * max_num_threads];
    short** rows0_t     = new short*[max_num_threads];
    short** rows1_t     = new short*[max_num_threads];
    int* prev_sy        = new int[max_num_threads];

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < max_num_threads; ++t) {
            prev_sy[t] = -2;
            rows0_t[t] = rows0 + t * (w * 3 + 1);
            rows1_t[t] = rows1 + t * (w * 3 + 1);
        }

        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            int thread_id = OMP_TID_;
            ResizeBilinearOneRow<3>(param, thread_id, rows0_t, rows1_t, prev_sy, b, w, h, stride, dy);
        }
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
    delete[] rows0_t;
    delete[] rows1_t;
    delete[] prev_sy;
}

void ResizeBilinearC4Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride, uint8_t* dst, int w,
                          int h, int stride) {
    ResizeBilinearPreparation(4);

    ResizeBilinearKernelParm param(xofs, yofs, ialpha, ibeta, src, dst, src_plane, src_stride, schannel);

    // loop body
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    short* rows0        = new short[(w * 4) * max_num_threads];
    short* rows1        = new short[(w * 4) * max_num_threads];
    short** rows0_t     = new short*[max_num_threads];
    short** rows1_t     = new short*[max_num_threads];
    int* prev_sy        = new int[max_num_threads];

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < max_num_threads; ++t) {
            prev_sy[t] = -2;
            rows0_t[t] = rows0 + t * (w * 4);
            rows1_t[t] = rows1 + t * (w * 4);
        }

        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            int thread_id = OMP_TID_;
            ResizeBilinearOneRow<4>(param, thread_id, rows0_t, rows1_t, prev_sy, b, w, h, stride, dy);
        }
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
    delete[] rows0_t;
    delete[] rows1_t;
    delete[] prev_sy;
}

void ResizeBilinearC1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return ResizeBilinearC1Impl(src, batch, src_w, src_h, src_w, dst, w, h, w);
}

void ResizeBilinearC2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return ResizeBilinearC2Impl(src, batch, src_w, src_h, src_w * 2, dst, w, h, w * 2);
}

void ResizeBilinearC3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return ResizeBilinearC3Impl(src, batch, src_w, src_h, src_w * 3, dst, w, h, w * 3);
}

void ResizeBilinearC4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return ResizeBilinearC4Impl(src, batch, src_w, src_h, src_w * 4, dst, w, h, w * 4);
}

void ResizeBilinearYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {
    // assert src_w % 2 == 0
    // assert src_h % 2 == 0
    // assert w % 2 == 0
    // assert h % 2 == 0

    int src_plane = src_w * src_h * 3 / 2;
    int dst_plane = w * h * 3 / 2;

    for (int b = 0; b < batch; ++b) {
        const uint8_t* srcY = src + b * src_plane;
        uint8_t* dstY       = dst + b * dst_plane;
        ResizeBilinearC1(srcY, 1, src_w, src_h, dstY, w, h);

        const uint8_t* srcUV = srcY + src_w * src_h;
        uint8_t* dstUV       = dstY + w * h;
        ResizeBilinearC2(srcUV, 1, src_w / 2, src_h / 2, dstUV, w / 2, h / 2);
    }
}

#define ResizeNearestPreparation(channel)                                                                              \
    int schannel = channel;                                                                                            \
    int* buf     = nullptr;                                                                                            \
    GetResizeBufNearset(src_w, src_h, w, h, schannel, &buf);                                                           \
    int* xofs       = buf;                                                                                             \
    int* yofs       = buf + w;                                                                                         \
    uint8_t* ialpha = (uint8_t*)(buf + w + h);                                                                         \
    uint8_t* ibeta  = (uint8_t*)(buf + w + h + w);

#define ResizeNearestLoopPreparation()                                                                                 \
    int sy            = (ibeta[dy] == 0) ? yofs[dy] + 1 : yofs[dy];                                                    \
    const uint8_t* Sp = src + src_stride * (b * src_h + sy);                                                           \
    uint8_t* Dp       = dst + stride * (b * h + dy);                                                                   \
    int dx            = 0;

void ResizeNearestC1Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride, uint8_t* dst, int w,
                         int h, int stride) {
    ResizeNearestPreparation(1);

    // loop body
    for (int b = 0; b < batch; ++b) {
        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            ResizeNearestLoopPreparation();
#ifdef __SSE4_2__
            int* xofs_p       = xofs;
            uint8_t* ialpha_p = ialpha;
            uint8_t* Dp_p     = Dp;
            __m128i _S0, _tmp0, _tmp1;
            __m128i _mask0 = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
            __m128i _mask1 = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14);
            int simd_loop  = 0;
            for (int i = 0; i < w - 7; i += 8) {
                __m128i _mask = _mm_loadl_epi64((__m128i*)ialpha_p);  // 01234567

                _S0 = _mm_insert_epi16(_S0, *(short*)(Sp + xofs_p[0]), 0);
                _S0 = _mm_insert_epi16(_S0, *(short*)(Sp + xofs_p[1]), 1);
                _S0 = _mm_insert_epi16(_S0, *(short*)(Sp + xofs_p[2]), 2);
                _S0 = _mm_insert_epi16(_S0, *(short*)(Sp + xofs_p[3]), 3);
                _S0 = _mm_insert_epi16(_S0, *(short*)(Sp + xofs_p[4]), 4);
                _S0 = _mm_insert_epi16(_S0, *(short*)(Sp + xofs_p[5]), 5);
                _S0 = _mm_insert_epi16(_S0, *(short*)(Sp + xofs_p[6]), 6);
                _S0 = _mm_insert_epi16(_S0, *(short*)(Sp + xofs_p[7]), 7);  // 0l0r 1l1r ... 7l7r

                _tmp0 = _mm_shuffle_epi8(_S0, _mask0);
                _tmp1 = _mm_shuffle_epi8(_S0, _mask1);

                _mm_storel_epi64((__m128i*)Dp_p, _mm_blendv_epi8(_tmp1, _tmp0, _mask));

                xofs_p += 8;
                ialpha_p += 8;
                Dp_p += 8 * 1;
                ++simd_loop;
            }
            dx += simd_loop * 8;
#endif
            for (; dx < w; dx++) {
                int sx = xofs[dx];
                Dp[dx] = (ialpha[dx] == 0) ? Sp[sx + 1] : Sp[sx];
            }
        }
    }

    delete[] buf;
}

void ResizeNearestC2Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride, uint8_t* dst, int w,
                         int h, int stride) {
    ResizeNearestPreparation(2);

    // loop body
    for (int b = 0; b < batch; ++b) {
        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            ResizeNearestLoopPreparation();
#ifdef __SSE4_2__
            int* xofs_p       = xofs;
            uint8_t* ialpha_p = ialpha;
            uint8_t* Dp_p     = Dp;
            __m128i _S0, _S1, _tmp0, _tmp1;
            __m128i _mask0 = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
            int simd_loop  = 0;
            for (int i = 0; i < w - 7; i += 8) {
                __m128i _mask = _mm_loadl_epi64((__m128i*)ialpha_p);  // 01234567
                _mask         = _mm_unpacklo_epi8(_mask, _mask);      // 0011223344556677

                _S0 = _mm_insert_epi32(_S0, *(int*)(Sp + xofs_p[0]), 0);
                _S0 = _mm_insert_epi32(_S0, *(int*)(Sp + xofs_p[1]), 1);
                _S0 = _mm_insert_epi32(_S0, *(int*)(Sp + xofs_p[2]), 2);
                _S0 = _mm_insert_epi32(_S0, *(int*)(Sp + xofs_p[3]), 3);  // 0l0l0r0r 1l1l1r1r 2l2l2r2r 3l3l3r3r
                _S1 = _mm_insert_epi32(_S1, *(int*)(Sp + xofs_p[4]), 0);
                _S1 = _mm_insert_epi32(_S1, *(int*)(Sp + xofs_p[5]), 1);
                _S1 = _mm_insert_epi32(_S1, *(int*)(Sp + xofs_p[6]), 2);
                _S1 = _mm_insert_epi32(_S1, *(int*)(Sp + xofs_p[7]), 3);  // 4l4l4r4r 5l5l5r5r 6l6l6r6r 7l7l7r7r

                _tmp0 = _mm_shuffle_epi8(_S0, _mask0);  // 0l0l 1l1l 2l2l 3l3l 0r0r 1r1r 2r2r 3r3r
                _tmp1 = _mm_shuffle_epi8(_S1, _mask0);  // 4l4l 5l5l 6l6l 7l7l 4r4r 5r5r 6r6r 7r7r

                _S0 = _mm_unpacklo_epi64(_tmp0, _tmp1);  // 0l0l - 7l7l
                _S1 = _mm_unpackhi_epi64(_tmp0, _tmp1);  // 0r0r - 7r7r

                _mm_storeu_si128((__m128i*)Dp_p, _mm_blendv_epi8(_S1, _S0, _mask));

                xofs_p += 8;
                ialpha_p += 8;
                Dp_p += 8 * 2;
                ++simd_loop;
            }
            dx += simd_loop * 8;
#endif
            for (; dx < w; dx++) {
                int sx         = xofs[dx];
                Dp[dx * 2]     = (ialpha[dx] == 0) ? Sp[sx + 2] : Sp[sx];
                Dp[dx * 2 + 1] = (ialpha[dx] == 0) ? Sp[sx + 3] : Sp[sx + 1];
            }
        }
    }

    delete[] buf;
}

void ResizeNearestC3Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride, uint8_t* dst, int w,
                         int h, int stride) {
    ResizeNearestPreparation(3);

    // loop body
    for (int b = 0; b < batch; ++b) {
        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            ResizeNearestLoopPreparation();
#ifdef __SSE4_2__
            int* xofs_p       = xofs;
            uint8_t* ialpha_p = ialpha;
            uint8_t* Dp_p     = Dp;
            __m128i _S0, _S1, _S2, _S3, _tmp0, _tmp1;
            __m128i _mask0, _mask1;
            __m128i _mask_res = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15);
            int simd_loop     = 0;
            for (int i = 0; i < (w - 2) - 7; i += 8) {
                __m128i _mask = _mm_loadl_epi64((__m128i*)ialpha_p);  // 01234567
                _mask         = _mm_unpacklo_epi8(_mask, _mask);      // 0011223344556677
                _mask0        = _mm_unpacklo_epi8(_mask, _mask);      // 0000111122223333
                _mask1        = _mm_unpackhi_epi8(_mask, _mask);      // 4444555566667777

                _S0 = load_element_c3_pack4(Sp + xofs_p[0]);  // 0l0l0lxx0r0r0rxx
                _S1 = load_element_c3_pack4(Sp + xofs_p[1]);  // 1l1l1lxx1r1r1rxx
                _S2 = load_element_c3_pack4(Sp + xofs_p[2]);  // 2l2l2lxx2r2r2rxx
                _S3 = load_element_c3_pack4(Sp + xofs_p[3]);  // 3l3l3lxx3r3r3rxx

                _S0   = _mm_unpacklo_epi64(_S0, _S1);      // 0l 0r 1l 1r
                _S1   = _mm_unpacklo_epi64(_S2, _S3);      // 2l 2r 3l 3r
                _tmp0 = _mm_unpacklo_epi32(_S0, _S1);      // 0l 2l 0r 2r
                _tmp1 = _mm_unpackhi_epi32(_S0, _S1);      // 1l 3l 1r 3r
                _S0   = _mm_unpacklo_epi32(_tmp0, _tmp1);  // 0l 1l 2l 3l
                _S1   = _mm_unpackhi_epi32(_tmp0, _tmp1);  // 0r 1r 2r 3r

                _mm_storeu_si128((__m128i*)Dp_p, _mm_shuffle_epi8(_mm_blendv_epi8(_S1, _S0, _mask0), _mask_res));

                _S0 = load_element_c3_pack4(Sp + xofs_p[4]);  // 4l 4r
                _S1 = load_element_c3_pack4(Sp + xofs_p[5]);  // 5l 5r
                _S2 = load_element_c3_pack4(Sp + xofs_p[6]);  // 6l 6r
                _S3 = load_element_c3_pack4(Sp + xofs_p[7]);  // 7l 7r

                _S0   = _mm_unpacklo_epi64(_S0, _S1);      // 4l 4r 5l 5r
                _S1   = _mm_unpacklo_epi64(_S2, _S3);      // 6l 6r 7l 7r
                _tmp0 = _mm_unpacklo_epi32(_S0, _S1);      // 4l 6l 4r 6r
                _tmp1 = _mm_unpackhi_epi32(_S0, _S1);      // 5l 7l 5r 7r
                _S0   = _mm_unpacklo_epi32(_tmp0, _tmp1);  // 4l 5l 6l 7l
                _S1   = _mm_unpackhi_epi32(_tmp0, _tmp1);  // 4r 5r 6r 7r

                _mm_storeu_si128((__m128i*)(Dp_p + 12), _mm_shuffle_epi8(_mm_blendv_epi8(_S1, _S0, _mask1), _mask_res));

                xofs_p += 8;
                ialpha_p += 8;
                Dp_p += 8 * 3;
                ++simd_loop;
            }
            dx += simd_loop * 8;
#endif
            for (; dx < w; dx++) {
                int sx         = xofs[dx];
                Dp[dx * 3]     = (ialpha[dx] == 0) ? Sp[sx + 3] : Sp[sx];
                Dp[dx * 3 + 1] = (ialpha[dx] == 0) ? Sp[sx + 4] : Sp[sx + 1];
                Dp[dx * 3 + 2] = (ialpha[dx] == 0) ? Sp[sx + 5] : Sp[sx + 2];
            }
        }
    }

    delete[] buf;
}

void ResizeNearestC4Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride, uint8_t* dst, int w,
                         int h, int stride) {
    ResizeNearestPreparation(4);

    // loop body
    for (int b = 0; b < batch; ++b) {
        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            ResizeNearestLoopPreparation();
#ifdef __SSE4_2__
            int* xofs_p       = xofs;
            uint8_t* ialpha_p = ialpha;
            uint8_t* Dp_p     = Dp;
            __m128i _S0, _S1, _S2, _S3, _tmp0, _tmp1;
            __m128i _mask0, _mask1;
            int simd_loop = 0;
            for (int i = 0; i < w - 7; i += 8) {
                __m128i _mask = _mm_loadl_epi64((__m128i*)ialpha_p);  // 01234567
                _mask         = _mm_unpacklo_epi8(_mask, _mask);      // 0011223344556677
                _mask0        = _mm_unpacklo_epi8(_mask, _mask);      // 0000111122223333
                _mask1        = _mm_unpackhi_epi8(_mask, _mask);      // 4444555566667777

                _S0   = _mm_loadl_epi64((__m128i*)(Sp + xofs_p[0]));  // 0l0l0l0l0r0r0r0r
                _S1   = _mm_loadl_epi64((__m128i*)(Sp + xofs_p[1]));  // 1l1l1l1l1r1r1r1r
                _S2   = _mm_loadl_epi64((__m128i*)(Sp + xofs_p[2]));  // 2l2l2l2l2r2r2r2r
                _S3   = _mm_loadl_epi64((__m128i*)(Sp + xofs_p[3]));  // 3l3l3l3l3r3r3r3r
                _S0   = _mm_unpacklo_epi64(_S0, _S1);                 // 0l0l0l0l0r0r0r0r 1l1l1l1l1r1r1r1r
                _S1   = _mm_unpacklo_epi64(_S2, _S3);                 // 2l2l2l2l2r2r2r2r 3l3l3l3l3r3r3r3r
                _tmp0 = _mm_unpacklo_epi32(_S0, _S1);                 // 0l 2l 0r 2r
                _tmp1 = _mm_unpackhi_epi32(_S0, _S1);                 // 1l 3l 1r 3r
                _S0   = _mm_unpacklo_epi32(_tmp0, _tmp1);             // 0l 1l 2l 3l
                _S1   = _mm_unpackhi_epi32(_tmp0, _tmp1);             // 0r 1r 2r 3r
                _mm_storeu_si128((__m128i*)Dp_p, _mm_blendv_epi8(_S1, _S0, _mask0));

                _S0   = _mm_loadl_epi64((__m128i*)(Sp + xofs_p[4]));  // 4l 4r
                _S1   = _mm_loadl_epi64((__m128i*)(Sp + xofs_p[5]));  // 5l 5r
                _S2   = _mm_loadl_epi64((__m128i*)(Sp + xofs_p[6]));  // 6l 6r
                _S3   = _mm_loadl_epi64((__m128i*)(Sp + xofs_p[7]));  // 7l 7r
                _S0   = _mm_unpacklo_epi64(_S0, _S1);                 // 4l 4r 5l 5r
                _S1   = _mm_unpacklo_epi64(_S2, _S3);                 // 6l 6r 7l 7r
                _tmp0 = _mm_unpacklo_epi32(_S0, _S1);                 // 4l 6l 4r 6r
                _tmp1 = _mm_unpackhi_epi32(_S0, _S1);                 // 5l 7l 5r 7r
                _S0   = _mm_unpacklo_epi32(_tmp0, _tmp1);             // 4l 5l 6l 7l
                _S1   = _mm_unpackhi_epi32(_tmp0, _tmp1);             // 4r 5r 6r 7r
                _mm_storeu_si128((__m128i*)(Dp_p + 16), _mm_blendv_epi8(_S1, _S0, _mask1));

                xofs_p += 8;
                ialpha_p += 8;
                Dp_p += 8 * 4;
                ++simd_loop;
            }
            dx += simd_loop * 8;
#endif
            for (; dx < w; dx++) {
                int sx         = xofs[dx];
                Dp[dx * 4]     = (ialpha[dx] == 0) ? Sp[sx + 4] : Sp[sx];
                Dp[dx * 4 + 1] = (ialpha[dx] == 0) ? Sp[sx + 5] : Sp[sx + 1];
                Dp[dx * 4 + 2] = (ialpha[dx] == 0) ? Sp[sx + 6] : Sp[sx + 2];
                Dp[dx * 4 + 3] = (ialpha[dx] == 0) ? Sp[sx + 7] : Sp[sx + 3];
            }
        }
    }

    delete[] buf;
}

void ResizeNearestC1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return ResizeNearestC1Impl(src, batch, src_w, src_h, src_w, dst, w, h, w);
}

void ResizeNearestC2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return ResizeNearestC2Impl(src, batch, src_w, src_h, src_w * 2, dst, w, h, w * 2);
}

void ResizeNearestC3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return ResizeNearestC3Impl(src, batch, src_w, src_h, src_w * 3, dst, w, h, w * 3);
}

void ResizeNearestC4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return ResizeNearestC4Impl(src, batch, src_w, src_h, src_w * 4, dst, w, h, w * 4);
}

void ResizeNearestYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {
    // assert src_w % 2 == 0
    // assert src_h % 2 == 0
    // assert w % 2 == 0
    // assert h % 2 == 0

    int src_plane = src_w * src_h * 3 / 2;
    int dst_plane = w * h * 3 / 2;

    for (int b = 0; b < batch; ++b) {
        const uint8_t* srcY = src + b * src_plane;
        uint8_t* dstY       = dst + b * dst_plane;
        ResizeNearestC1(srcY, 1, src_w, src_h, dstY, w, h);

        const uint8_t* srcUV = srcY + src_w * src_h;
        uint8_t* dstUV       = dstY + w * h;
        ResizeNearestC2(srcUV, 1, src_w / 2, src_h / 2, dstUV, w / 2, h / 2);
    }
}

#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)
#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define KSIZE 2
static short BilinearTab_i[INTER_TAB_SIZE * INTER_TAB_SIZE][KSIZE][KSIZE];

// Interpolation table of size 32 x 32 x 4:
// (1*1,     0*1,     1*0,     0*0)    , ... , (1/32*1,     31/32*1,     1/32*0,     31/32*0)
// (1*31/32, 0*31/32, 1*1/32,  0*1/32) , ... , (1/32*31/32, 31/32*31/32, 1/32*1/32,  31/32*1/32)
//                                       ...
// (1*1/32,  0*1/32,  1*31/32, 0*31/32), ... , (1/32*1/32,  31/32*1/32,  1/32*31/32, 31/32*31/32)
static void InitInterTab2D() {
    static bool inited = false;
    if (inited) {
        return;
    }

    short* itab = BilinearTab_i[0][0];
    int ksize   = KSIZE;

    float* _tab = new float[2 * INTER_TAB_SIZE];
    int i, j, k1, k2;
    InitInterTab1D(_tab, INTER_TAB_SIZE);
    for (i = 0; i < INTER_TAB_SIZE; i++) {
        for (j = 0; j < INTER_TAB_SIZE; j++, itab += ksize * ksize) {
            int isum = 0;

            for (k1 = 0; k1 < ksize; k1++) {
                float vy = _tab[i * ksize + k1];
                for (k2 = 0; k2 < ksize; k2++) {
                    float v                       = vy * _tab[j * ksize + k2];
                    isum += itab[k1 * ksize + k2] = SATURATE_CAST_SHORT(v * INTER_REMAP_COEF_SCALE);
                }
            }

            if (isum != INTER_REMAP_COEF_SCALE) {
                int diff   = isum - INTER_REMAP_COEF_SCALE;
                int ksize2 = ksize / 2, Mk1 = ksize2, Mk2 = ksize2, mk1 = ksize2, mk2 = ksize2;
                for (k1 = ksize2; k1 < ksize2 + 2; k1++)
                    for (k2 = ksize2; k2 < ksize2 + 2; k2++) {
                        if (itab[k1 * ksize + k2] < itab[mk1 * ksize + mk2])
                            mk1 = k1, mk2 = k2;
                        else if (itab[k1 * ksize + k2] > itab[Mk1 * ksize + Mk2])
                            Mk1 = k1, Mk2 = k2;
                    }
                if (diff < 0)
                    itab[Mk1 * ksize + Mk2] = (short)(itab[Mk1 * ksize + Mk2] - diff);
                else
                    itab[mk1 * ksize + mk2] = (short)(itab[mk1 * ksize + mk2] - diff);
            }
        }
    }

    delete[] _tab;
}

// The buffer contains adelta and bdelta, which are used to calculate src position (src_x, src_y)
// from dst position (x, y):
// src_x = adelta[2*x]   + bdelta[2*y]
// src_y = adelta[2*x+1] + bdelta[2*y+1]
static void WarpAffineInit(uint8_t* dst, int batch, int dst_w, int dst_h, int channel, const float border_val,
                           const float (*transform)[3], int** buffer) {
    uint8_t border_ival = (uint8_t)border_val;
    memset(dst, border_ival, batch * dst_h * dst_w * channel);

    // Init LookUp Table
    InitInterTab2D();

    double m[6];
    WarpAffineMatrixInverse(transform, m);

    *buffer = reinterpret_cast<int*>(x86Malloc((dst_w + dst_h) * 2 * sizeof(int)));

    int* adelta = *buffer;
    int* bdelta = *buffer + dst_w * 2;

    for (int x = 0; x < dst_w; x++) {
        *adelta++ = SATURATE_CAST_INT(m[0] * x * 1024);
        *adelta++ = SATURATE_CAST_INT(m[3] * x * 1024);
    }

    for (int y = 0; y < dst_h; y++) {
        *bdelta++ = SATURATE_CAST_INT((m[1] * y + m[2]) * 1024);
        *bdelta++ = SATURATE_CAST_INT((m[4] * y + m[5]) * 1024);
    }
}

static inline bool CheckDataIsOnBoundary(const int new_x_loc, const int new_y_loc, const int src_w, const int src_h) {
    return new_x_loc >= -1 && new_x_loc <= (src_w - 1) && new_y_loc >= -1 && new_y_loc <= (src_h - 1);
}

static inline bool CheckDataIsInBoundary(const int new_x_loc, const int new_y_loc, const int src_w, const int src_h) {
    return new_x_loc >= 0 && new_x_loc < (src_w - 1) && new_y_loc >= 0 && new_y_loc < (src_h - 1);
}

static void WarpAffinePrepareOneRow(int* buf_loc, short* tab_loc, int* adelta, int* bdelta, int channel,
                                    const uint8_t* src, int src_w, int src_h, uint8_t* dst, int dst_w, int y,
                                    int src_offset, int& x_count, int& end_x, float border_val = 0) {
    const unsigned char* src2 = src + src_w * channel;

    short* xy_loc_buf = new short[dst_w * 2 + 4];
    short* tb_loc_buf = new short[dst_w + 4];
    // short xy_loc_buf[dst_w * 2];
    // short tb_loc_buf[dst_w];
    // int   sc_loc_buf[dst_w];
    // short* xy_loc_buf_p = xy_loc_buf;
    // short* tb_loc_buf_p = tb_loc_buf;
    int x = 0;
#if defined(__SSE4_2__)
    {
        __m128i off_vec   = _mm_set1_epi32(16);
        __m128i mask31    = _mm_set1_epi32(31);
        __m128i mask_mull = _mm_setr_epi16(1, 32, 1, 32, 1, 32, 1, 32);

        __m128i bdelta_vec = _mm_setr_epi32(bdelta[2 * y], bdelta[2 * y + 1], bdelta[2 * y], bdelta[2 * y + 1]);
        for (; x < dst_w; x += 4) {
            __m128i adelta0 = _mm_add_epi32(_mm_loadu_si128((__m128i*)(adelta + 2 * x)), off_vec);
            __m128i adelta1 = _mm_add_epi32(_mm_loadu_si128((__m128i*)(adelta + 2 * x + 4)), off_vec);
            // x0y0,x1y1
            __m128i x0y0 = _mm_add_epi32(adelta0, bdelta_vec);
            // x2y2,x3y3
            __m128i x2y2     = _mm_add_epi32(adelta1, bdelta_vec);
            x0y0             = _mm_srai_epi32(x0y0, 5);
            x2y2             = _mm_srai_epi32(x2y2, 5);
            __m128i xy_float = _mm_packs_epi32(_mm_and_si128(x0y0, mask31), _mm_and_si128(x2y2, mask31));
            xy_float         = _mm_mullo_epi16(xy_float, mask_mull);
            __m128i xy       = _mm_packs_epi32(_mm_srai_epi32(x0y0, 5), _mm_srai_epi32(x2y2, 5));
            xy_float         = _mm_hadd_epi16(xy_float, xy_float);
            _mm_storeu_si128((__m128i*)(xy_loc_buf + x * 2), xy);
            _mm_storel_epi64((__m128i*)(tb_loc_buf + x), xy_float);
        }
    }
#endif
    for (; x < dst_w; ++x) {
        int new_x             = adelta[2 * x] + bdelta[2 * y] + 16;
        int new_y             = adelta[2 * x + 1] + bdelta[2 * y + 1] + 16;
        int new_x_loc         = new_x >> 10;
        int new_y_loc         = new_y >> 10;
        xy_loc_buf[2 * x]     = new_x_loc;
        xy_loc_buf[2 * x + 1] = new_y_loc;
        tb_loc_buf[x]         = ((new_x >> 5) & 31) + ((new_y >> 5) & 31) * 32;
        // sc_loc_buf[x] = (new_x_loc + new_y_loc * src_w) * channel + src_offset;
    }

    for (x = 0; x < dst_w; ++x) {
        short new_x_loc    = xy_loc_buf[2 * x];
        short new_y_loc    = xy_loc_buf[2 * x + 1];
        short new_xy_float = tb_loc_buf[x];
        int src_loc        = (new_x_loc + new_y_loc * src_w) * channel + src_offset;

        if ((unsigned)new_x_loc < (src_w - 1) && (unsigned)new_y_loc < (src_h - 1)) {
            buf_loc[x] = src_loc;
            tab_loc[x] = new_xy_float;
            x_count++;
            end_x = x;
        } else if (CheckDataIsOnBoundary(new_x_loc, new_y_loc, src_w, src_h)) {
            short* wtab = BilinearTab_i[new_xy_float][0];
            int dsc_loc = x * channel;

            int mask0 = new_x_loc >= 0 && new_y_loc >= 0;
            int mask1 = new_x_loc <= (src_w - 2) && new_y_loc >= 0;
            int mask2 = new_x_loc >= 0 && new_y_loc <= (src_h - 2);
            int mask3 = new_x_loc <= (src_w - 2) && new_y_loc <= (src_h - 2);

            for (int c = 0; c < channel; ++c) {
                int val_xy = 0;
                val_xy += wtab[0] * (mask0 ? src[src_loc + c] : border_val);
                val_xy += wtab[1] * (mask1 ? src[src_loc + channel + c] : border_val);
                val_xy += wtab[2] * (mask2 ? src2[src_loc + c] : border_val);
                val_xy += wtab[3] * (mask3 ? src2[src_loc + channel + c] : border_val);
                dst[dsc_loc + c] = SATURATE_CAST_UCHAR((val_xy + (1 << 14)) >> 15);
            }
        }
    }

    delete[] xy_loc_buf;
    delete[] tb_loc_buf;
}

template <int schannel>
static void WarpAffineCalculateOneRow(int begin_x, int end_x, int channel, int dst_loc_base, const int* buf_loc,
                                      const short* tab_loc, const uint8_t* src1, const uint8_t* src2, uint8_t* dst) {
    const int* buf_loc_p   = buf_loc + begin_x;
    const short* tab_loc_p = tab_loc + begin_x;
    const short* tab_p     = BilinearTab_i[0][0];
    int x                  = begin_x;

#if defined(__SSE4_2__)
    __m128i mask_vec;
    if (channel == 4) {
        mask_vec = _mm_setr_epi8(0, 4, 1, 5, 2, 6, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    } else if (channel == 3) {
        mask_vec = _mm_setr_epi8(0, 3, 1, 4, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    } else if (channel == 2) {
        mask_vec = _mm_setr_epi8(0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    }

    const int shift_l = (4 - schannel) * 4;
    const int shift_r = 4 - schannel;
    __m128i DELTA_vec = _mm_set1_epi32(1 << 14);
    __m128i mask_vec2 = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);
    __m128i mask_vec3 = _mm_setr_epi8(4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7);

    uint8_t* dst_loc_p = dst + dst_loc_base + x * channel;
    for (; x + 8 <= end_x - 2 * shift_r; x += 8) {
        short* wtab0 = BilinearTab_i[tab_loc[x]][0];
        short* wtab1 = BilinearTab_i[tab_loc[x + 1]][0];
        short* wtab2 = BilinearTab_i[tab_loc[x + 2]][0];
        short* wtab3 = BilinearTab_i[tab_loc[x + 3]][0];
        short* wtab4 = BilinearTab_i[tab_loc[x + 4]][0];
        short* wtab5 = BilinearTab_i[tab_loc[x + 5]][0];
        short* wtab6 = BilinearTab_i[tab_loc[x + 6]][0];
        short* wtab7 = BilinearTab_i[tab_loc[x + 7]][0];

        const uint8_t* src_base0  = src1 + buf_loc[x];
        const uint8_t* src_base1  = src1 + buf_loc[x + 1];
        const uint8_t* src_base2  = src1 + buf_loc[x + 2];
        const uint8_t* src_base3  = src1 + buf_loc[x + 3];
        const uint8_t* src_base4  = src1 + buf_loc[x + 4];
        const uint8_t* src_base5  = src1 + buf_loc[x + 5];
        const uint8_t* src_base6  = src1 + buf_loc[x + 6];
        const uint8_t* src_base7  = src1 + buf_loc[x + 7];
        const uint8_t* src2_base0 = src2 + buf_loc[x];
        const uint8_t* src2_base1 = src2 + buf_loc[x + 1];
        const uint8_t* src2_base2 = src2 + buf_loc[x + 2];
        const uint8_t* src2_base3 = src2 + buf_loc[x + 3];
        const uint8_t* src2_base4 = src2 + buf_loc[x + 4];
        const uint8_t* src2_base5 = src2 + buf_loc[x + 5];
        const uint8_t* src2_base6 = src2 + buf_loc[x + 6];
        const uint8_t* src2_base7 = src2 + buf_loc[x + 7];

        __m128i wtab0_vec = _mm_loadl_epi64((__m128i*)wtab0);
        __m128i wtab1_vec = _mm_loadl_epi64((__m128i*)wtab1);
        __m128i wtab2_vec = _mm_loadl_epi64((__m128i*)wtab2);
        __m128i wtab3_vec = _mm_loadl_epi64((__m128i*)wtab3);
        __m128i wtab4_vec = _mm_loadl_epi64((__m128i*)wtab4);
        __m128i wtab5_vec = _mm_loadl_epi64((__m128i*)wtab5);
        __m128i wtab6_vec = _mm_loadl_epi64((__m128i*)wtab6);
        __m128i wtab7_vec = _mm_loadl_epi64((__m128i*)wtab7);

        __m128i wtab00_vec = _mm_shuffle_epi8(wtab0_vec, mask_vec2);
        __m128i wtab01_vec = _mm_shuffle_epi8(wtab0_vec, mask_vec3);
        __m128i wtab10_vec = _mm_shuffle_epi8(wtab1_vec, mask_vec2);
        __m128i wtab11_vec = _mm_shuffle_epi8(wtab1_vec, mask_vec3);
        __m128i wtab20_vec = _mm_shuffle_epi8(wtab2_vec, mask_vec2);
        __m128i wtab21_vec = _mm_shuffle_epi8(wtab2_vec, mask_vec3);
        __m128i wtab30_vec = _mm_shuffle_epi8(wtab3_vec, mask_vec2);
        __m128i wtab31_vec = _mm_shuffle_epi8(wtab3_vec, mask_vec3);
        __m128i wtab40_vec = _mm_shuffle_epi8(wtab4_vec, mask_vec2);
        __m128i wtab41_vec = _mm_shuffle_epi8(wtab4_vec, mask_vec3);
        __m128i wtab50_vec = _mm_shuffle_epi8(wtab5_vec, mask_vec2);
        __m128i wtab51_vec = _mm_shuffle_epi8(wtab5_vec, mask_vec3);
        __m128i wtab60_vec = _mm_shuffle_epi8(wtab6_vec, mask_vec2);
        __m128i wtab61_vec = _mm_shuffle_epi8(wtab6_vec, mask_vec3);
        __m128i wtab70_vec = _mm_shuffle_epi8(wtab7_vec, mask_vec2);
        __m128i wtab71_vec = _mm_shuffle_epi8(wtab7_vec, mask_vec3);

        __m128i point_vec00;
        __m128i point_vec01;
        __m128i point_vec10;
        __m128i point_vec11;
        __m128i point_vec20;
        __m128i point_vec21;
        __m128i point_vec30;
        __m128i point_vec31;
        __m128i point_vec40;
        __m128i point_vec41;
        __m128i point_vec50;
        __m128i point_vec51;
        __m128i point_vec60;
        __m128i point_vec61;
        __m128i point_vec70;
        __m128i point_vec71;

        if (schannel == 3) {
            point_vec00 = _mm_shuffle_epi8(load_element_c3(src_base0), mask_vec);
            point_vec01 = _mm_shuffle_epi8(load_element_c3(src2_base0), mask_vec);
            point_vec10 = _mm_shuffle_epi8(load_element_c3(src_base1), mask_vec);
            point_vec11 = _mm_shuffle_epi8(load_element_c3(src2_base1), mask_vec);
            point_vec20 = _mm_shuffle_epi8(load_element_c3(src_base2), mask_vec);
            point_vec21 = _mm_shuffle_epi8(load_element_c3(src2_base2), mask_vec);
            point_vec30 = _mm_shuffle_epi8(load_element_c3(src_base3), mask_vec);
            point_vec31 = _mm_shuffle_epi8(load_element_c3(src2_base3), mask_vec);
            point_vec40 = _mm_shuffle_epi8(load_element_c3(src_base4), mask_vec);
            point_vec41 = _mm_shuffle_epi8(load_element_c3(src2_base4), mask_vec);
            point_vec50 = _mm_shuffle_epi8(load_element_c3(src_base5), mask_vec);
            point_vec51 = _mm_shuffle_epi8(load_element_c3(src2_base5), mask_vec);
            point_vec60 = _mm_shuffle_epi8(load_element_c3(src_base6), mask_vec);
            point_vec61 = _mm_shuffle_epi8(load_element_c3(src2_base6), mask_vec);
            point_vec70 = _mm_shuffle_epi8(load_element_c3(src_base7), mask_vec);
            point_vec71 = _mm_shuffle_epi8(load_element_c3(src2_base7), mask_vec);
        } else if (schannel == 4) {
            point_vec00 = _mm_shuffle_epi8(load_element_c4(src_base0), mask_vec);
            point_vec01 = _mm_shuffle_epi8(load_element_c4(src2_base0), mask_vec);
            point_vec10 = _mm_shuffle_epi8(load_element_c4(src_base1), mask_vec);
            point_vec11 = _mm_shuffle_epi8(load_element_c4(src2_base1), mask_vec);
            point_vec20 = _mm_shuffle_epi8(load_element_c4(src_base2), mask_vec);
            point_vec21 = _mm_shuffle_epi8(load_element_c4(src2_base2), mask_vec);
            point_vec30 = _mm_shuffle_epi8(load_element_c4(src_base3), mask_vec);
            point_vec31 = _mm_shuffle_epi8(load_element_c4(src2_base3), mask_vec);
            point_vec40 = _mm_shuffle_epi8(load_element_c4(src_base4), mask_vec);
            point_vec41 = _mm_shuffle_epi8(load_element_c4(src2_base4), mask_vec);
            point_vec50 = _mm_shuffle_epi8(load_element_c4(src_base5), mask_vec);
            point_vec51 = _mm_shuffle_epi8(load_element_c4(src2_base5), mask_vec);
            point_vec60 = _mm_shuffle_epi8(load_element_c4(src_base6), mask_vec);
            point_vec61 = _mm_shuffle_epi8(load_element_c4(src2_base6), mask_vec);
            point_vec70 = _mm_shuffle_epi8(load_element_c4(src_base7), mask_vec);
            point_vec71 = _mm_shuffle_epi8(load_element_c4(src2_base7), mask_vec);
        } else if (schannel == 2) {
            point_vec00 = _mm_shuffle_epi8(load_element_c2(src_base0), mask_vec);
            point_vec01 = _mm_shuffle_epi8(load_element_c2(src2_base0), mask_vec);
            point_vec10 = _mm_shuffle_epi8(load_element_c2(src_base1), mask_vec);
            point_vec11 = _mm_shuffle_epi8(load_element_c2(src2_base1), mask_vec);
            point_vec20 = _mm_shuffle_epi8(load_element_c2(src_base2), mask_vec);
            point_vec21 = _mm_shuffle_epi8(load_element_c2(src2_base2), mask_vec);
            point_vec30 = _mm_shuffle_epi8(load_element_c2(src_base3), mask_vec);
            point_vec31 = _mm_shuffle_epi8(load_element_c2(src2_base3), mask_vec);
            point_vec40 = _mm_shuffle_epi8(load_element_c2(src_base4), mask_vec);
            point_vec41 = _mm_shuffle_epi8(load_element_c2(src2_base4), mask_vec);
            point_vec50 = _mm_shuffle_epi8(load_element_c2(src_base5), mask_vec);
            point_vec51 = _mm_shuffle_epi8(load_element_c2(src2_base5), mask_vec);
            point_vec60 = _mm_shuffle_epi8(load_element_c2(src_base6), mask_vec);
            point_vec61 = _mm_shuffle_epi8(load_element_c2(src2_base6), mask_vec);
            point_vec70 = _mm_shuffle_epi8(load_element_c2(src_base7), mask_vec);
            point_vec71 = _mm_shuffle_epi8(load_element_c2(src2_base7), mask_vec);
        }

        point_vec00 = _mm_cvtepu8_epi16(point_vec00);
        point_vec01 = _mm_cvtepu8_epi16(point_vec01);
        point_vec10 = _mm_cvtepu8_epi16(point_vec10);
        point_vec11 = _mm_cvtepu8_epi16(point_vec11);
        point_vec20 = _mm_cvtepu8_epi16(point_vec20);
        point_vec21 = _mm_cvtepu8_epi16(point_vec21);
        point_vec30 = _mm_cvtepu8_epi16(point_vec30);
        point_vec31 = _mm_cvtepu8_epi16(point_vec31);
        point_vec40 = _mm_cvtepu8_epi16(point_vec40);
        point_vec41 = _mm_cvtepu8_epi16(point_vec41);
        point_vec50 = _mm_cvtepu8_epi16(point_vec50);
        point_vec51 = _mm_cvtepu8_epi16(point_vec51);
        point_vec60 = _mm_cvtepu8_epi16(point_vec60);
        point_vec61 = _mm_cvtepu8_epi16(point_vec61);
        point_vec70 = _mm_cvtepu8_epi16(point_vec70);
        point_vec71 = _mm_cvtepu8_epi16(point_vec71);

        // int val_xy0  = wtab[0] * point0 + wtab[1] * point1 + wtab[2] * point2 + wtab[3] * point3;
        // dst[dst_loc] = SATURATE_CAST_UCHAR((val_xy0 + (1 << 14)) >> 15);
        __m128i p0 = _mm_add_epi32(_mm_madd_epi16(wtab00_vec, point_vec00), DELTA_vec);
        __m128i p1 = _mm_add_epi32(_mm_madd_epi16(wtab10_vec, point_vec10), DELTA_vec);
        __m128i p2 = _mm_add_epi32(_mm_madd_epi16(wtab20_vec, point_vec20), DELTA_vec);
        __m128i p3 = _mm_add_epi32(_mm_madd_epi16(wtab30_vec, point_vec30), DELTA_vec);
        __m128i p4 = _mm_add_epi32(_mm_madd_epi16(wtab40_vec, point_vec40), DELTA_vec);
        __m128i p5 = _mm_add_epi32(_mm_madd_epi16(wtab50_vec, point_vec50), DELTA_vec);
        __m128i p6 = _mm_add_epi32(_mm_madd_epi16(wtab60_vec, point_vec60), DELTA_vec);
        __m128i p7 = _mm_add_epi32(_mm_madd_epi16(wtab70_vec, point_vec70), DELTA_vec);
        p0         = _mm_add_epi32(_mm_madd_epi16(wtab01_vec, point_vec01), p0);
        p1         = _mm_add_epi32(_mm_madd_epi16(wtab11_vec, point_vec11), p1);
        p2         = _mm_add_epi32(_mm_madd_epi16(wtab21_vec, point_vec21), p2);
        p3         = _mm_add_epi32(_mm_madd_epi16(wtab31_vec, point_vec31), p3);
        p4         = _mm_add_epi32(_mm_madd_epi16(wtab41_vec, point_vec41), p4);
        p5         = _mm_add_epi32(_mm_madd_epi16(wtab51_vec, point_vec51), p5);
        p6         = _mm_add_epi32(_mm_madd_epi16(wtab61_vec, point_vec61), p6);
        p7         = _mm_add_epi32(_mm_madd_epi16(wtab71_vec, point_vec71), p7);
        p0         = _mm_srli_epi32(p0, 15);
        p1         = _mm_srli_epi32(p1, 15);
        p2         = _mm_srli_epi32(p2, 15);
        p3         = _mm_srli_epi32(p3, 15);
        p4         = _mm_srli_epi32(p4, 15);
        p5         = _mm_srli_epi32(p5, 15);
        p6         = _mm_srli_epi32(p6, 15);
        p7         = _mm_srli_epi32(p7, 15);

        // store
        __m128i ans0 = _mm_packs_epi32(_mm_slli_si128(p0, shift_l), p1);
        __m128i ans1 = _mm_packs_epi32(_mm_slli_si128(p2, shift_l), p3);
        __m128i ans2 = _mm_packs_epi32(_mm_slli_si128(p4, shift_l), p5);
        __m128i ans3 = _mm_packs_epi32(_mm_slli_si128(p6, shift_l), p7);
        ans0         = _mm_packus_epi16(ans0, ans0);
        ans1         = _mm_packus_epi16(ans1, ans1);
        ans2         = _mm_packus_epi16(ans2, ans2);
        ans3         = _mm_packus_epi16(ans3, ans3);
        ans0         = _mm_srli_si128(ans0, shift_r);
        ans1         = _mm_srli_si128(ans1, shift_r);
        ans2         = _mm_srli_si128(ans2, shift_r);
        ans3         = _mm_srli_si128(ans3, shift_r);
        _mm_storel_epi64((__m128i*)dst_loc_p, ans0);
        _mm_storel_epi64((__m128i*)(dst_loc_p + channel * 2), ans1);
        _mm_storel_epi64((__m128i*)(dst_loc_p + channel * 4), ans2);
        _mm_storel_epi64((__m128i*)(dst_loc_p + channel * 6), ans3);

        dst_loc_p += 8 * channel;
    }
#endif

    if (channel == 2) {
        for (; x <= end_x; x++) {
            int dst_loc = dst_loc_base + x * 2;
            int src_loc = buf_loc[x];
            short* wtab = BilinearTab_i[tab_loc[x]][0];

            int point00 = src1[src_loc];
            int point01 = src1[src_loc + 1];
            int point02 = src1[src_loc + 2];
            int point03 = src1[src_loc + 3];
            int point10 = src2[src_loc];
            int point11 = src2[src_loc + 1];
            int point12 = src2[src_loc + 2];
            int point13 = src2[src_loc + 3];

            int val_xy0      = wtab[0] * point00 + wtab[1] * point02 + wtab[2] * point10 + wtab[3] * point12;
            int val_xy1      = wtab[0] * point01 + wtab[1] * point03 + wtab[2] * point11 + wtab[3] * point13;
            dst[dst_loc]     = SATURATE_CAST_UCHAR((val_xy0 + (1 << 14)) >> 15);
            dst[dst_loc + 1] = SATURATE_CAST_UCHAR((val_xy1 + (1 << 14)) >> 15);
        }
    } else if (channel == 3) {
        for (; x <= end_x; x++) {
            int dst_loc = dst_loc_base + x * 3;
            int src_loc = buf_loc[x];
            short* wtab = BilinearTab_i[tab_loc[x]][0];

            int point00 = src1[src_loc];
            int point01 = src1[src_loc + 1];
            int point02 = src1[src_loc + 2];
            int point03 = src1[src_loc + 3];
            int point04 = src1[src_loc + 4];
            int point05 = src1[src_loc + 5];
            int point10 = src2[src_loc];
            int point11 = src2[src_loc + 1];
            int point12 = src2[src_loc + 2];
            int point13 = src2[src_loc + 3];
            int point14 = src2[src_loc + 4];
            int point15 = src2[src_loc + 5];

            int val_xy0      = wtab[0] * point00 + wtab[1] * point03 + wtab[2] * point10 + wtab[3] * point13;
            int val_xy1      = wtab[0] * point01 + wtab[1] * point04 + wtab[2] * point11 + wtab[3] * point14;
            int val_xy2      = wtab[0] * point02 + wtab[1] * point05 + wtab[2] * point12 + wtab[3] * point15;
            dst[dst_loc]     = SATURATE_CAST_UCHAR((val_xy0 + (1 << 14)) >> 15);
            dst[dst_loc + 1] = SATURATE_CAST_UCHAR((val_xy1 + (1 << 14)) >> 15);
            dst[dst_loc + 2] = SATURATE_CAST_UCHAR((val_xy2 + (1 << 14)) >> 15);
        }
    } else if (channel == 4) {
        for (; x <= end_x; x++) {
            int dst_loc = dst_loc_base + x * 4;
            int src_loc = buf_loc[x];
            short* wtab = BilinearTab_i[tab_loc[x]][0];

            int point00 = src1[src_loc];
            int point01 = src1[src_loc + 1];
            int point02 = src1[src_loc + 2];
            int point03 = src1[src_loc + 3];
            int point04 = src1[src_loc + 4];
            int point05 = src1[src_loc + 5];
            int point06 = src1[src_loc + 6];
            int point07 = src1[src_loc + 7];
            int point10 = src2[src_loc];
            int point11 = src2[src_loc + 1];
            int point12 = src2[src_loc + 2];
            int point13 = src2[src_loc + 3];
            int point14 = src2[src_loc + 4];
            int point15 = src2[src_loc + 5];
            int point16 = src2[src_loc + 6];
            int point17 = src2[src_loc + 7];

            int val_xy0      = wtab[0] * point00 + wtab[1] * point04 + wtab[2] * point10 + wtab[3] * point14;
            int val_xy1      = wtab[0] * point01 + wtab[1] * point05 + wtab[2] * point11 + wtab[3] * point15;
            int val_xy2      = wtab[0] * point02 + wtab[1] * point06 + wtab[2] * point12 + wtab[3] * point16;
            int val_xy3      = wtab[0] * point03 + wtab[1] * point07 + wtab[2] * point13 + wtab[3] * point17;
            dst[dst_loc]     = SATURATE_CAST_UCHAR((val_xy0 + (1 << 14)) >> 15);
            dst[dst_loc + 1] = SATURATE_CAST_UCHAR((val_xy1 + (1 << 14)) >> 15);
            dst[dst_loc + 2] = SATURATE_CAST_UCHAR((val_xy2 + (1 << 14)) >> 15);
            dst[dst_loc + 3] = SATURATE_CAST_UCHAR((val_xy3 + (1 << 14)) >> 15);
        }
    }
}

template <>
void WarpAffineCalculateOneRow<1>(int begin_x, int end_x, int channel, int dst_loc_base, const int* buf_loc,
                                  const short* tab_loc, const uint8_t* src1, const uint8_t* src2, uint8_t* dst) {
    const int* buf_loc_p   = buf_loc + begin_x;
    const short* tab_loc_p = tab_loc + begin_x;
    const short* tab_p     = BilinearTab_i[0][0];
    int x                  = begin_x;

    // uint8_t buf[(end_x - begin_x + 4) * 4];
    uint8_t* buf = new uint8_t[(end_x - begin_x + 4) * 4];
    uint8_t* ptr = buf;

    for (int x = begin_x; x <= end_x; x++) {
        int src_loc = buf_loc[x];
        *ptr++      = src1[src_loc];
        *ptr++      = src1[src_loc + 1];
        *ptr++      = src2[src_loc];
        *ptr++      = src2[src_loc + 1];
    }
    ptr = buf;

#if defined(__SSE4_2__)
    __m128i DELTA_vec  = _mm_set1_epi32(1 << 14);
    uint8_t* dst_loc_p = dst + dst_loc_base + x * channel;
    for (; x + 7 <= end_x; x += 8) {
        short* wtab0 = BilinearTab_i[tab_loc[x]][0];
        short* wtab1 = BilinearTab_i[tab_loc[x + 1]][0];
        short* wtab2 = BilinearTab_i[tab_loc[x + 2]][0];
        short* wtab3 = BilinearTab_i[tab_loc[x + 3]][0];
        short* wtab4 = BilinearTab_i[tab_loc[x + 4]][0];
        short* wtab5 = BilinearTab_i[tab_loc[x + 5]][0];
        short* wtab6 = BilinearTab_i[tab_loc[x + 6]][0];
        short* wtab7 = BilinearTab_i[tab_loc[x + 7]][0];

        __m128i p0_load = _mm_loadl_epi64((__m128i*)ptr);
        __m128i p1_load = _mm_loadl_epi64((__m128i*)(ptr + 8));
        __m128i p2_load = _mm_loadl_epi64((__m128i*)(ptr + 16));
        __m128i p3_load = _mm_loadl_epi64((__m128i*)(ptr + 24));

        __m128i wvec0 = _mm_loadl_epi64((__m128i*)wtab0);
        __m128i wvec1 = _mm_loadl_epi64((__m128i*)wtab1);
        __m128i wvec2 = _mm_loadl_epi64((__m128i*)wtab2);
        __m128i wvec3 = _mm_loadl_epi64((__m128i*)wtab3);
        __m128i wvec4 = _mm_loadl_epi64((__m128i*)wtab4);
        __m128i wvec5 = _mm_loadl_epi64((__m128i*)wtab5);
        __m128i wvec6 = _mm_loadl_epi64((__m128i*)wtab6);
        __m128i wvec7 = _mm_loadl_epi64((__m128i*)wtab7);

        __m128i w0 = _mm_unpacklo_epi64(wvec0, wvec1);
        __m128i w1 = _mm_unpacklo_epi64(wvec2, wvec3);
        __m128i w2 = _mm_unpacklo_epi64(wvec4, wvec5);
        __m128i w3 = _mm_unpacklo_epi64(wvec6, wvec7);

        __m128i p0 = _mm_cvtepu8_epi16(p0_load);
        __m128i p1 = _mm_cvtepu8_epi16(p1_load);
        __m128i p2 = _mm_cvtepu8_epi16(p2_load);
        __m128i p3 = _mm_cvtepu8_epi16(p3_load);

        __m128i r0 = _mm_madd_epi16(p0, w0);
        __m128i r1 = _mm_madd_epi16(p1, w1);
        __m128i r2 = _mm_madd_epi16(p2, w2);
        __m128i r3 = _mm_madd_epi16(p3, w3);

        __m128i ans0 = _mm_hadd_epi32(r0, r1);
        __m128i ans1 = _mm_hadd_epi32(r2, r3);

        ans0 = _mm_srai_epi32(_mm_add_epi32(ans0, DELTA_vec), 15);
        ans1 = _mm_srai_epi32(_mm_add_epi32(ans1, DELTA_vec), 15);

        __m128i ans_16 = _mm_packs_epi32(ans0, ans1);
        __m128i ans_8  = _mm_packus_epi16(ans_16, ans_16);
        _mm_storel_epi64((__m128i*)dst_loc_p, ans_8);
        dst_loc_p += 8;
        ptr += 32;
    }
#endif
    for (; x <= end_x; x++) {
        int dst_loc = dst_loc_base + x * 1;
        int src_loc = buf_loc[x];
        short* wtab = BilinearTab_i[tab_loc[x]][0];

        int point0 = ptr[0];
        int point1 = ptr[1];
        int point2 = ptr[2];
        int point3 = ptr[3];
        ptr += 4;

        int val_xy0  = wtab[0] * point0 + wtab[1] * point1 + wtab[2] * point2 + wtab[3] * point3;
        dst[dst_loc] = SATURATE_CAST_UCHAR((val_xy0 + (1 << 14)) >> 15);
    }

    delete[] buf;
}

template <int schannel>
static void WarpAffineBilinear(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                               const float (*transform)[3], const float border_val) {
    int src_plane = src_h * src_w * schannel;

    int* buffer = nullptr;
    WarpAffineInit(dst, batch, dst_w, dst_h, schannel, border_val, transform, &buffer);
    int* adelta = buffer;
    int* bdelta = buffer + dst_w * 2;

    int max_num_threads = OMP_MAX_THREADS_NUM_;
    int* buf_loc        = new int[dst_w * max_num_threads];
    short* tab_loc      = new short[dst_w * max_num_threads];

    const unsigned char* src2 = src + src_w * schannel;

    OMP_PARALLEL_FOR_
    for (int y = 0; y < dst_h * batch; ++y) {
        int thread_id    = OMP_TID_;
        int x_count      = 0;
        int end_x        = 0;
        int dst_loc_base = y * dst_w * schannel;
        int* buf_loc_t   = buf_loc + thread_id * dst_w;
        short* tab_loc_t = tab_loc + thread_id * dst_w;

        WarpAffinePrepareOneRow(buf_loc_t, tab_loc_t, adelta, bdelta, schannel, src, src_w, src_h, dst + dst_loc_base,
                                dst_w, y % dst_h, (y / dst_h) * src_plane, x_count, end_x, border_val);
        WarpAffineCalculateOneRow<schannel>(end_x - x_count + 1, end_x, schannel, dst_loc_base, buf_loc_t, tab_loc_t,
                                            src, src2, dst);
    }

    delete[] buf_loc;
    delete[] tab_loc;

    x86Free(buffer);
}

// warp affine
void WarpAffineBilinearC1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                          const float (*transform)[3], const float border_val) {
    WarpAffineBilinear<1>(src, batch, src_w, src_h, dst, dst_w, dst_h, transform, border_val);
}

void WarpAffineBilinearC2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                          const float (*transform)[3], const float border_val) {
    WarpAffineBilinear<2>(src, batch, src_w, src_h, dst, dst_w, dst_h, transform, border_val);
}

void WarpAffineBilinearC3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                          const float (*transform)[3], const float border_val) {
    WarpAffineBilinear<3>(src, batch, src_w, src_h, dst, dst_w, dst_h, transform, border_val);
}

void WarpAffineBilinearC4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                          const float (*transform)[3], const float border_val) {
    WarpAffineBilinear<4>(src, batch, src_w, src_h, dst, dst_w, dst_h, transform, border_val);
}

void WarpAffineBilinearYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                                const float (*transform)[3], const float border_val) {
    // assert src_w % 2 == 0
    // assert src_h % 2 == 0
    // assert dst_w % 2 == 0
    // assert dst_h % 2 == 0

    int src_plane = src_w * src_h * 3 / 2;
    int dst_plane = dst_w * dst_h * 3 / 2;

    for (int b = 0; b < batch; ++b) {
        const uint8_t* srcY = src + b * src_plane;
        uint8_t* dstY       = dst + b * dst_plane;
        WarpAffineBilinearC1(srcY, 1, src_w, src_h, dstY, dst_w, dst_h, transform, border_val);

        const uint8_t* srcUV = srcY + src_w * src_h;
        uint8_t* dstUV       = dstY + dst_w * dst_h;
        WarpAffineBilinearC2(srcUV, 1, src_w / 2, src_h / 2, dstUV, dst_w / 2, dst_h / 2, transform, border_val);
    }
}

template <int schannel>
static void WarpAffineNearest(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                              const float (*transform)[3], const float border_val) {
    uint8_t border_ival = (uint8_t)border_val;
    int* buffer         = nullptr;
    WarpAffineInit(dst, batch, dst_w, dst_h, schannel, border_val, transform, &buffer);
    int* adelta = buffer;
    int* bdelta = buffer + dst_w * 2;

    int src_stride = src_w * schannel;
    int src_plane  = src_h * src_w * schannel;
    OMP_PARALLEL_FOR_
    for (int y = 0; y < dst_h * batch; ++y) {
        int y_c = y / dst_h;
        int y_r = y % dst_h;

        auto src_b = src + y_c * src_plane;
        auto dst_y = dst + y * dst_w * schannel;

        for (int x = 0; x < dst_w; ++x) {
            int new_x     = adelta[2 * x] + bdelta[2 * y_r] + 16;
            int new_y     = adelta[2 * x + 1] + bdelta[2 * y_r + 1] + 16;
            int new_x_loc = new_x >> 10;
            int new_y_loc = new_y >> 10;

            bool is_left = ((new_x >> 5) & 31) < 16;
            bool is_top  = ((new_y >> 5) & 31) < 16;

            int src_loc = (new_x_loc + new_y_loc * src_w) * schannel;
            auto src_y1 = src_b + src_loc;
            auto src_y2 = src_y1 + src_stride;
            auto dst_x  = dst_y + x * schannel;

            if (CheckDataIsInBoundary(new_x_loc, new_y_loc, src_w, src_h)) {
                int c = 0;
#ifdef __SSE4_2__
                if (schannel == 4) {
                    __m128i _vsrc    = is_top ? _mm_loadl_epi64((__m128i*)src_y1) : _mm_loadl_epi64((__m128i*)src_y2);
                    *(int32_t*)dst_x = is_left ? _mm_extract_epi32(_vsrc, 0) : _mm_extract_epi32(_vsrc, 1);
                    c                = 4;
                }
#endif
                for (; c < schannel; c++) {
                    uint8_t point00 = src_y1[c];
                    uint8_t point01 = src_y1[schannel + c];
                    uint8_t point10 = src_y2[c];
                    uint8_t point11 = src_y2[schannel + c];
                    if (is_top) {
                        dst_x[c] = is_left ? point00 : point01;
                    } else {
                        dst_x[c] = is_left ? point10 : point11;
                    }
                }
            } else if (CheckDataIsOnBoundary(new_x_loc, new_y_loc, src_w, src_h)) {
                int mask0 = new_x_loc >= 0 && new_y_loc >= 0;
                int mask1 = new_x_loc <= (src_w - 2) && new_y_loc >= 0;
                int mask2 = new_x_loc >= 0 && new_y_loc <= (src_h - 2);
                int mask3 = new_x_loc <= (src_w - 2) && new_y_loc <= (src_h - 2);

                for (int c = 0; c < schannel; ++c) {
                    uint8_t point00 = mask0 ? src_y1[c] : border_ival;
                    uint8_t point01 = mask1 ? src_y1[schannel + c] : border_ival;
                    uint8_t point10 = mask2 ? src_y2[c] : border_ival;
                    uint8_t point11 = mask3 ? src_y2[schannel + c] : border_ival;
                    if (is_top) {
                        dst_x[c] = is_left ? point00 : point01;
                    } else {
                        dst_x[c] = is_left ? point10 : point11;
                    }
                }
            }
        }
    }

    free(buffer);
}

void WarpAffineNearestC1(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                         const float (*transform)[3], const float border_val) {
    WarpAffineNearest<1>(src, batch, src_w, src_h, dst, dst_w, dst_h, transform, border_val);
}

void WarpAffineNearestC2(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                         const float (*transform)[3], const float border_val) {
    WarpAffineNearest<2>(src, batch, src_w, src_h, dst, dst_w, dst_h, transform, border_val);
}

void WarpAffineNearestC3(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                         const float (*transform)[3], const float border_val) {
    WarpAffineNearest<3>(src, batch, src_w, src_h, dst, dst_w, dst_h, transform, border_val);
}

void WarpAffineNearestC4(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                         const float (*transform)[3], const float border_val) {
    WarpAffineNearest<4>(src, batch, src_w, src_h, dst, dst_w, dst_h, transform, border_val);
}

void WarpAffineNearestYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                               const float (*transform)[3], const float border_val) {
    // assert src_w % 2 == 0
    // assert src_h % 2 == 0
    // assert dst_w % 2 == 0
    // assert dst_h % 2 == 0

    int src_plane = src_w * src_h * 3 / 2;
    int dst_plane = dst_w * dst_h * 3 / 2;

    for (int b = 0; b < batch; ++b) {
        const uint8_t* srcY = src + b * src_plane;
        uint8_t* dstY       = dst + b * dst_plane;
        WarpAffineNearestC1(srcY, 1, src_w, src_h, dstY, dst_w, dst_h, transform, border_val);

        const uint8_t* srcUV = srcY + src_w * src_h;
        uint8_t* dstUV       = dstY + dst_w * dst_h;
        WarpAffineNearestC2(srcUV, 1, src_w / 2, src_h / 2, dstUV, dst_w / 2, dst_h / 2, transform, border_val);
    }
}

}  // namespace x86
}  // namespace TNN_NS
