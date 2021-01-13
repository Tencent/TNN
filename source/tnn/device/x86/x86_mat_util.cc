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

#include "tnn/device/x86/x86_common.h"
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

    short *xy_loc_buf = new short[dst_w * 2 + 4];
    short *tb_loc_buf = new short[dst_w + 4];
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
    for (; x + 8 <= end_x; x += 8) {
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

        __m128i point_vec00 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src_base0), mask_vec);
        __m128i point_vec01 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src2_base0), mask_vec);
        __m128i point_vec10 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src_base1), mask_vec);
        __m128i point_vec11 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src2_base1), mask_vec);
        __m128i point_vec20 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src_base2), mask_vec);
        __m128i point_vec21 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src2_base2), mask_vec);
        __m128i point_vec30 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src_base3), mask_vec);
        __m128i point_vec31 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src2_base3), mask_vec);
        __m128i point_vec40 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src_base4), mask_vec);
        __m128i point_vec41 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src2_base4), mask_vec);
        __m128i point_vec50 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src_base5), mask_vec);
        __m128i point_vec51 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src2_base5), mask_vec);
        __m128i point_vec60 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src_base6), mask_vec);
        __m128i point_vec61 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src2_base6), mask_vec);
        __m128i point_vec70 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src_base7), mask_vec);
        __m128i point_vec71 = _mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)src2_base7), mask_vec);

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
    uint8_t *buf = new uint8_t[(end_x - begin_x + 4) * 4];
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
