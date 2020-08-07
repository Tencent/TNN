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

#include "tnn/device/arm/arm_mat_util.h"

#include <type_traits>

#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

#include "tnn/core/macro.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), 0), UCHAR_MAX)
#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X) (int)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

void mat_memcpy_2d(void* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    auto src_ptr = reinterpret_cast<uint8_t*>(src);
    auto dst_ptr = reinterpret_cast<uint8_t*>(dst);

    for (int h = 0; h < height; h++) {
        memcpy(dst_ptr, src_ptr, width);
        src_ptr += src_stride;
        dst_ptr += dst_stride;
    }

}

static void calculate_position_and_ratio(int length, double scale, int border, int channel,
                                         int* position, short* ratio) {
    const int INTER_RESIZE_COEF_BITS  = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    float pos_f;
    float rat_f;
    int pos_i;
    for (int i = 0; i < length; i++) {
        pos_f = (float)((i + 0.5) * scale - 0.5);
        pos_i = static_cast<int>(floor(pos_f));
        rat_f = pos_f - pos_i;

        if (pos_i < 0) {
            pos_i = 0;
            rat_f = 0.f;
        }
        if (pos_i >= border - 1) {
            pos_i = border - 2;
            rat_f = 1.f;
        }

        position[i] = pos_i * channel;

        float a0 = (1.f - rat_f) * INTER_RESIZE_COEF_SCALE;
        float a1 = rat_f * INTER_RESIZE_COEF_SCALE;

        ratio[i * 2]     = SATURATE_CAST_SHORT(a0);
        ratio[i * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }
}

// Meanings of xofs, yofs, ialpha, ibeta in src image:
//                               |  ialpha[2*x]  |  ialpha[2*x+1]  |
//     --       (xofs[x], yofs[y])                                 (xofs[x]+1, yofs[y])
// ibeta[2*y]
//     --                              (x*scale_x, y*scale_y)
// ibeta[2*y+1]
//     --       (xofs[x], yofs[y]+1)                               (xofs[x]+1, yofs[y]+1)
static void get_resize_buf(int src_w, int src_h, int w, int h, int c, int** buf) {
    const int INTER_RESIZE_COEF_BITS  = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

    double scale_x = (double)src_w / w;
    double scale_y = (double)src_h / h;

    *buf = new int[w + h + w + h];

    int* xofs = *buf;
    int* yofs = *buf + w;

    short* ialpha = (short*)(*buf + w + h);
    short* ibeta  = (short*)(*buf + w + h + w);

    calculate_position_and_ratio(w, scale_x, src_w, c, xofs, ialpha);
    calculate_position_and_ratio(h, scale_y, src_h, 1, yofs, ibeta);
}

static void get_adjacent_rows(int sy, int prev_sy, short** rows0, short** rows1, int* xofs, 
                              const uint8_t* src, int src_stride, int c, int w, const short* ialphap) {
    if (sy == prev_sy) {
        // reuse all rows
    } else if (sy == prev_sy + 1) {
        // hresize one row
        short* rows0_old  = *rows0;
        *rows0            = *rows1;
        *rows1            = rows0_old;
        const uint8_t* S1 = src + src_stride * (sy + 1);

        short* rows1p        = *rows1;
        for (int dx = 0; dx < w; dx++) {
            int sx   = xofs[dx];
            short a0 = ialphap[0];
            short a1 = ialphap[1];

            const uint8_t* S1p = S1 + sx;

#ifndef TNN_USE_NEON
            for (int dc = 0; dc < c; ++dc) {
                rows1p[dc]         = (S1p[dc] * a0 + S1p[dc + c] * a1) >> 4;
            }
#else
            if (c == 2) {
                int16x4_t _a0a1XX = vld1_s16(ialphap);
                int16x4_t _a0a0a1a1 = vzip_s16(_a0a1XX, _a0a1XX).val[0];
                uint8x8_t _S1 = uint8x8_t();

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                _S1 = vld1_lane_u8(S1p + 3, _S1, 3);

                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1lowhigh = vget_low_s16(_S116);
                int32x4_t _S1ma0a1 = vmull_s16(_S1lowhigh, _a0a0a1a1);
                int32x2_t _rows1low = vadd_s32(vget_low_s32(_S1ma0a1), vget_high_s32(_S1ma0a1));
                int32x4_t _rows1 = vcombine_s32(_rows1low, vget_high_s32(_S1ma0a1));
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
            } else if (c == 3) {
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S1 = uint8x8_t();

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                _S1 = vld1_lane_u8(S1p + 3, _S1, 3);
                _S1 = vld1_lane_u8(S1p + 4, _S1, 4);
                _S1 = vld1_lane_u8(S1p + 5, _S1, 5);

                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S1high = vext_s16(_S1low, vget_high_s16(_S116), 3);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
            } else if (c == 4) {
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S1 = vld1_u8(S1p);

                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S1high = vget_high_s16(_S116);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
            } else {
                for (int dc = 0; dc < c; ++dc) {
                    rows1p[dc]         = (S1p[dc] * a0 + S1p[dc + c] * a1) >> 4;
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

        short* rows0p        = *rows0;
        short* rows1p        = *rows1;
        for (int dx = 0; dx < w; dx++) {
            int sx   = xofs[dx];
            short a0 = ialphap[0];
            short a1 = ialphap[1];

            const uint8_t* S0p = S0 + sx;
            const uint8_t* S1p = S1 + sx;

#ifndef TNN_USE_NEON
            for (int dc = 0; dc < c; ++dc) {
                rows0p[dc]         = (S0p[dc] * a0 + S0p[dc + c] * a1) >> 4;
                rows1p[dc]         = (S1p[dc] * a0 + S1p[dc + c] * a1) >> 4;
            }
#else
            if (c == 2) {
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S0 = uint8x8_t();
                uint8x8_t _S1 = uint8x8_t();

                _S0 = vld1_lane_u8(S0p, _S0, 0);
                _S0 = vld1_lane_u8(S0p + 1, _S0, 1);
                _S0 = vld1_lane_u8(S0p + 2, _S0, 2);
                _S0 = vld1_lane_u8(S0p + 3, _S0, 3);

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                _S1 = vld1_lane_u8(S1p + 3, _S1, 3);

                int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0lowhigh = vget_low_s16(_S016);
                int16x4_t _S1lowhigh = vget_low_s16(_S116);
                int32x2x2_t _S0S1low_S0S1high = vtrn_s32(vreinterpret_s32_s16(_S0lowhigh), vreinterpret_s32_s16(_S1lowhigh));
                int32x4_t _rows01 = vmull_s16(vreinterpret_s16_s32(_S0S1low_S0S1high.val[0]), _a0);
                _rows01 = vmlal_s16(_rows01, vreinterpret_s16_s32(_S0S1low_S0S1high.val[1]), _a1);
                int16x4_t _rows01_sr4 = vshrn_n_s32(_rows01, 4);
                int16x4_t _rows1_sr4 = vext_s16(_rows01_sr4, _rows01_sr4, 2);
                vst1_s16(rows0p, _rows01_sr4);
                vst1_s16(rows1p, _rows1_sr4);
            } else if (c == 3) {
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S0 = uint8x8_t();
                uint8x8_t _S1 = uint8x8_t();

                _S0 = vld1_lane_u8(S0p, _S0, 0);
                _S0 = vld1_lane_u8(S0p + 1, _S0, 1);
                _S0 = vld1_lane_u8(S0p + 2, _S0, 2);
                _S0 = vld1_lane_u8(S0p + 3, _S0, 3);
                _S0 = vld1_lane_u8(S0p + 4, _S0, 4);
                _S0 = vld1_lane_u8(S0p + 5, _S0, 5);

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                _S1 = vld1_lane_u8(S1p + 3, _S1, 3);
                _S1 = vld1_lane_u8(S1p + 4, _S1, 4);
                _S1 = vld1_lane_u8(S1p + 5, _S1, 5);

                int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0low = vget_low_s16(_S016);
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S0high = vext_s16(_S0low, vget_high_s16(_S016), 3);
                int16x4_t _S1high = vext_s16(_S1low, vget_high_s16(_S116), 3);
                int32x4_t _rows0 = vmull_s16(_S0low, _a0);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows0 = vmlal_s16(_rows0, _S0high, _a1);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows0p, _rows0_sr4);
                vst1_s16(rows1p, _rows1_sr4);
            } else if (c == 4) {
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S0 = vld1_u8(S0p);
                uint8x8_t _S1 = vld1_u8(S1p);
                int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0low = vget_low_s16(_S016);
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S0high = vget_high_s16(_S016);
                int16x4_t _S1high = vget_high_s16(_S116);
                int32x4_t _rows0 = vmull_s16(_S0low, _a0);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows0 = vmlal_s16(_rows0, _S0high, _a1);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows0p, _rows0_sr4);
                vst1_s16(rows1p, _rows1_sr4);
            } else {
                for (int dc = 0; dc < c; ++dc) {
                    rows0p[dc]         = (S0p[dc] * a0 + S0p[dc + c] * a1) >> 4;
                    rows1p[dc]         = (S1p[dc] * a0 + S1p[dc + c] * a1) >> 4;
                }
            }
#endif

            ialphap += 2;
            rows0p += c;
            rows1p += c;
        }
    }
}

static void calculate_one_row(short* rows0p, short* rows1p, const int b0, const int b1, const int w, const int c,
                              uint8_t* Dp) {
#ifndef TNN_USE_NEON
    int remain = w * c;
#else
    int nn = (w * c) >> 3;
    int remain = (w * c) - (nn << 3);
    int16x4_t _b0 = vdup_n_s16(b0);
    int16x4_t _b1 = vdup_n_s16(b1);
    int32x4_t _v2 = vdupq_n_s32(2);
    for (; nn > 0; nn--) {
        int16x4_t _rows0p_sr4   = vld1_s16(rows0p);
        int16x4_t _rows1p_sr4   = vld1_s16(rows1p);
        int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p + 4);
        int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p + 4);

        int32x4_t _rows0p_sr4_mb0   = vmull_s16(_rows0p_sr4, _b0);
        int32x4_t _rows1p_sr4_mb1   = vmull_s16(_rows1p_sr4, _b1);
        int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
        int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

        int32x4_t _acc = _v2;
        _acc           = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
        _acc           = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

        int32x4_t _acc_1 = _v2;
        _acc_1           = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
        _acc_1           = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

        int16x4_t _acc16   = vshrn_n_s32(_acc, 2);
        int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

        uint8x8_t _D = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

        vst1_u8(Dp, _D);

        Dp += 8;
        rows0p += 8;
        rows1p += 8;
    }
#endif
    for (; remain; --remain) {
        *Dp++ = (uint8_t)(
            ((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
    }
}

void resize_bilinear_c1_impl(const uint8_t* src, int src_w, int src_h, int src_stride,
                             uint8_t* dst, int w, int h, int stride) {
    int* buf = nullptr;
    get_resize_buf(src_w, src_h, w, h, 1, &buf);
    int* xofs = buf;
    int* yofs = buf + w;
    short* ialpha = (short*)(buf + w + h);
    short* ibeta  = (short*)(buf + w + h + w);

    // loop body
    short* rows0 = new short[w];
    short* rows1 = new short[w];

    int prev_sy = -2;

    for (int dy = 0; dy < h; dy++) {
        int sy = yofs[dy];
        get_adjacent_rows(sy, prev_sy, &rows0, &rows1, xofs, src, src_stride, 1, w, ialpha);
        prev_sy = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        uint8_t* Dp   = dst + stride * (dy);

        calculate_one_row(rows0, rows1, b0, b1, w, 1, Dp);

        ibeta += 2;
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
}

void resize_bilinear_c2_impl(const uint8_t* src, int src_w, int src_h, int src_stride,
                             uint8_t* dst, int w, int h, int stride) {
    int* buf = nullptr;
    get_resize_buf(src_w, src_h, w, h, 2, &buf);
    int* xofs = buf;
    int* yofs = buf + w;
    short* ialpha = (short*)(buf + w + h);
    short* ibeta  = (short*)(buf + w + h + w);

    // loop body
    short* rows0 = new short[w * 2 + 2];
    short* rows1 = new short[w * 2 + 2];

    int prev_sy = -2;

    for (int dy = 0; dy < h; dy++) {
        int sy = yofs[dy];
        get_adjacent_rows(sy, prev_sy, &rows0, &rows1, xofs, src, src_stride, 2, w, ialpha);
        prev_sy = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        uint8_t* Dp   = dst + stride * (dy);

        calculate_one_row(rows0, rows1, b0, b1, w, 2, Dp);

        ibeta += 2;
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
}

void resize_bilinear_c3_impl(const uint8_t* src, int src_w, int src_h, int src_stride,
                             uint8_t* dst, int w, int h, int stride) {
    int* buf = nullptr;
    get_resize_buf(src_w, src_h, w, h, 3, &buf);
    int* xofs = buf;
    int* yofs = buf + w;
    short* ialpha = (short*)(buf + w + h);
    short* ibeta  = (short*)(buf + w + h + w);

    // loop body
    short* rows0 = new short[w * 3 + 1];
    short* rows1 = new short[w * 3 + 1];

    int prev_sy = -2;

    for (int dy = 0; dy < h; dy++) {
        int sy = yofs[dy];
        get_adjacent_rows(sy, prev_sy, &rows0, &rows1, xofs, src, src_stride, 3, w, ialpha);
        prev_sy = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        uint8_t* Dp   = dst + stride * (dy);

        calculate_one_row(rows0, rows1, b0, b1, w, 3, Dp);

        ibeta += 2;
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
}

void resize_bilinear_c4_impl(const uint8_t* src, int src_w, int src_h, int src_stride,
                             uint8_t* dst, int w, int h, int stride) {
    int* buf = nullptr;
    get_resize_buf(src_w, src_h, w, h, 4, &buf);
    int* xofs = buf;
    int* yofs = buf + w;
    short* ialpha = (short*)(buf + w + h);
    short* ibeta  = (short*)(buf + w + h + w);

    // loop body
    short* rows0 = new short[w * 4];
    short* rows1 = new short[w * 4];

    int prev_sy = -2;

    for (int dy = 0; dy < h; dy++) {
        int sy = yofs[dy];
        get_adjacent_rows(sy, prev_sy, &rows0, &rows1, xofs, src, src_stride, 4, w, ialpha);
        prev_sy = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        uint8_t* Dp   = dst + stride * (dy);

        calculate_one_row(rows0, rows1, b0, b1, w, 4, Dp);

        ibeta += 2;
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
}

void resize_bilinear_yuv420sp(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h) {
    // assert src_w % 2 == 0
    // assert src_h % 2 == 0
    // assert w % 2 == 0
    // assert h % 2 == 0

    const uint8_t* srcY = src;
    uint8_t* dstY       = dst;
    resize_bilinear_c1(srcY, src_w, src_h, dstY, w, h);

    const uint8_t* srcUV = src + src_w * src_h;
    uint8_t* dstUV       = dst + w * h;
    resize_bilinear_c2(srcUV, src_w / 2, src_h / 2, dstUV, w / 2, h / 2);
}

void resize_bilinear_c1(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return resize_bilinear_c1_impl(src, src_w, src_h, src_w, dst, w, h, w);
}

void resize_bilinear_c2(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return resize_bilinear_c2_impl(src, src_w, src_h, src_w * 2, dst, w, h, w * 2);
}

void resize_bilinear_c3(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return resize_bilinear_c3_impl(src, src_w, src_h, src_w * 3, dst, w, h, w * 3);
}

void resize_bilinear_c4(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return resize_bilinear_c4_impl(src, src_w, src_h, src_w * 4, dst, w, h, w * 4);
}


#define INTER_REMAP_COEF_BITS  15
#define INTER_REMAP_COEF_SCALE (1<<INTER_REMAP_COEF_BITS)
#define INTER_BITS      5
#define INTER_TAB_SIZE  (1<<INTER_BITS)
#define KSIZE 2
static short BilinearTab_i[INTER_TAB_SIZE*INTER_TAB_SIZE][KSIZE][KSIZE];

static inline void interpolateLinear(float x, float* coeffs) {
    coeffs[0] = 1.f - x;
    coeffs[1] = x;
}

static void initInterTab1D(float* tab, int tabsz) {
    float scale = 1.f / tabsz;
    for (int i = 0; i < tabsz; i++, tab += 2)
        interpolateLinear(i * scale, tab);
}

// Interpolation table of size 32 x 32 x 4:
// (1*1,     0*1,     1*0,     0*0)    , ... , (1/32*1,     31/32*1,     1/32*0,     31/32*0)
// (1*31/32, 0*31/32, 1*1/32,  0*1/32) , ... , (1/32*31/32, 31/32*31/32, 1/32*1/32,  31/32*1/32)
//                                       ...
// (1*1/32,  0*1/32,  1*31/32, 0*31/32), ... , (1/32*1/32,  31/32*1/32,  1/32*31/32, 31/32*31/32)
static void initInterTab2D() {
    static bool inited = false;
    if (inited) {
        return;
    }

    short* itab = BilinearTab_i[0][0];
    int ksize = KSIZE;

    float* _tab = new float[2 * INTER_TAB_SIZE];
    int i, j, k1, k2;
    initInterTab1D(_tab, INTER_TAB_SIZE);
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
static void warpaffine_init(uint8_t* dst, int dst_w, int dst_h, int channel, const float border_val,
                            const float (*transform)[3], int** buffer) {
    uint8_t border_ival = (uint8_t)border_val;
    if (border_ival) {
        for (int i = 0; i < dst_h * dst_w * channel; ++i) {
            dst[i] = border_ival;
        }
    } else {
        memset(dst, 0, dst_h * dst_w * channel);
    }

    // Init LookUp Table
    initInterTab2D();

    double m[6];
    double M[6];
    M[0] = transform[0][0];
    M[1] = transform[0][1];
    M[2] = transform[0][2];
    M[3] = transform[1][0];
    M[4] = transform[1][1];
    M[5] = transform[1][2];

    // Inverse transform matrix
    double D   = M[0] * M[4] - M[1] * M[3];
    D          = D != 0 ? 1. / D : 0;
    double A11 = M[4] * D, A22 = M[0] * D;
    m[0]      = A11;
    m[1]      = M[1] * (-D);
    m[3]      = M[3] * (-D);
    m[4]      = A22;
    double b1 = -A11 * M[2] - m[1] * M[5];
    double b2 = -m[3] * M[2] - A22 * M[5];
    m[2]      = b1;
    m[5]      = b2;

    int status = posix_memalign(reinterpret_cast<void**>(buffer), 32, (dst_w + dst_h) * 2 * sizeof(int));

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

void warpaffine_bilinear_c1(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                            const float (*transform)[3], const float border_val) {
    int* buffer = nullptr;
    warpaffine_init(dst, dst_w, dst_h, 1, border_val, transform, &buffer);

    int* adelta = buffer;
    int* bdelta = buffer + dst_w * 2;
    int DELTA = 1 << 14;

    int scols             = src_w;
    int srows             = src_h;
    int schannel          = 1;
    int stmp              = scols * schannel;
    unsigned int* buf_loc = new unsigned int[dst_w];
    short* tab_loc        = new short[dst_w];

    unsigned short* buf_point = (unsigned short*)buf_loc;
    const unsigned char* src2 = src + stmp;

    for (int y = 0; y < dst_h; ++y) {
        int x_count        = 0;
        int end_x          = 0;
        int final_loc_base = y * dst_w;
        for (int x = 0; x < dst_w; ++x) {
            int final_loc = final_loc_base + x;
            int new_x      = adelta[2 * x] + bdelta[2 * y] + 16;
            int new_y      = adelta[2 * x + 1] + bdelta[2 * y + 1] + 16;
            int new_x_full = new_x >> 5;
            int new_y_full = new_y >> 5;
            int new_x_loc  = new_x >> 10;
            int new_y_loc  = new_y >> 10;

            short new_xy_float = (new_x_full & 31) + (new_y_full & 31) * 32;
            short* wtab        = BilinearTab_i[new_xy_float][0];
            int loc_base       = new_y_loc * stmp + new_x_loc;

            if (new_x_loc >= -1 && new_x_loc <= (scols - 1) && new_y_loc >= -1 && new_y_loc <= (srows - 1)) {
                if ((unsigned)new_x_loc < (scols - 1) && (unsigned)new_y_loc < (srows - 1)) {
                    unsigned short* ptr  = (unsigned short*)(src + loc_base);
                    unsigned short* ptr2 = (unsigned short*)(src2 + loc_base);
                    buf_point[2 * x]     = ptr[0];
                    buf_point[2 * x + 1] = ptr2[0];
                    tab_loc[x]           = new_xy_float;
                    x_count++;
                    end_x = x;
                } else {
                    int mask0 =
                        new_x_loc >= 0 && new_x_loc <= (scols - 1) && new_y_loc >= 0 && new_y_loc <= (srows - 1);
                    int mask1 =
                        new_x_loc >= -1 && new_x_loc <= (scols - 2) && new_y_loc >= 0 && new_y_loc <= (srows - 1);
                    int mask2 =
                        new_x_loc >= 0 && new_x_loc <= (scols - 1) && new_y_loc >= -1 && new_y_loc <= (srows - 2);
                    int mask3 =
                        new_x_loc >= -1 && new_x_loc <= (scols - 2) && new_y_loc >= -1 && new_y_loc <= (srows - 2);
                    int val_xy0 = 0;

                    if (mask0) {
                        val_xy0 += wtab[0] * src[loc_base];
                    }
                    if (mask1) {
                        val_xy0 += wtab[1] * src[loc_base + 1];
                    }
                    if (mask2) {
                        val_xy0 += wtab[2] * src2[loc_base];
                    }
                    if (mask3) {
                        val_xy0 += wtab[3] * src2[loc_base + 1];
                    }
                    dst[final_loc] = SATURATE_CAST_UCHAR((val_xy0 + DELTA) >> 15);
                }
            }
        }

        int x      = end_x - x_count + 1;
        unsigned char* ptr = (unsigned char*)(buf_loc + x);

        for (; x <= end_x; x++) {
            int final_loc = final_loc_base + x;
            short* wtab   = BilinearTab_i[tab_loc[x]][0];

            int point0 = ptr[0];
            int point1 = ptr[1];
            int point2 = ptr[2];
            int point3 = ptr[3];
            ptr += 4;

            int val_xy0    = wtab[0] * point0 + wtab[1] * point1 + wtab[2] * point2 + wtab[3] * point3;
            dst[final_loc] = SATURATE_CAST_UCHAR((val_xy0 + DELTA) >> 15);
        }
    }
    delete[] buf_loc;
    delete[] tab_loc;

    free(buffer);
}

void warpaffine_bilinear_c3(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
                            const float (*transform)[3], const float border_val) {
    int* buffer = nullptr;
    warpaffine_init(dst, dst_w, dst_h, 3, border_val, transform, &buffer);

    int* adelta = buffer;
    int* bdelta = buffer + dst_w * 2;
    int DELTA = 1 << 14;

    int scols      = src_w;
    int srows      = src_h;
    int schannel   = 3;
    int stmp       = scols * schannel;
    int* buf_loc   = new int[dst_w + 4];
    short* tab_loc = new short[dst_w + 4];

    const unsigned char* src2 = src + stmp;

    short xy_loc_buf[dst_w * 2];
    short xy_float_buf[dst_w];

    for (int y = 0; y < dst_h; ++y) {
        int x_count        = 0;
        int end_x          = 0;
        int final_loc_base = y * dst_w * 3;

        for (int x = 0; x < dst_w; ++x) {
            int new_x               = adelta[2 * x] + bdelta[2 * y] + 16;
            int new_y               = adelta[2 * x + 1] + bdelta[2 * y + 1] + 16;
            int new_x_full          = new_x >> 5;
            int new_y_full          = new_y >> 5;
            xy_loc_buf[x * 2]     = (new_x >> 10);
            xy_loc_buf[x * 2 + 1] = (new_y >> 10);
            xy_float_buf[x]       = (new_x_full & 31) + (new_y_full & 31) * 32;

            int new_x_loc    = xy_loc_buf[x * 2];
            int new_y_loc    = xy_loc_buf[x * 2 + 1];
            int new_xy_float = xy_float_buf[x];
            short* wtab      = BilinearTab_i[new_xy_float][0];

            if ((unsigned)new_x_loc < (scols - 1) && (unsigned)new_y_loc < (srows - 1)) {
                buf_loc[x] = new_x_loc * 3 + new_y_loc * stmp;
                tab_loc[x] = new_xy_float;
                x_count++;
                end_x = x;
            } else {
                if (new_x_loc >= -1 && new_x_loc <= (scols - 1) && new_y_loc >= -1 && new_y_loc <= (srows - 1)) {
                    int loc_buffer = new_x_loc * 3 + new_y_loc * stmp;
                    int final_loc  = final_loc_base + x * 3;

                    int mask0 =
                        new_x_loc >= 0 && new_x_loc <= (scols - 1) && new_y_loc >= 0 && new_y_loc <= (srows - 1);
                    int mask1 =
                        new_x_loc >= -1 && new_x_loc <= (scols - 2) && new_y_loc >= 0 && new_y_loc <= (srows - 1);
                    int mask2 =
                        new_x_loc >= 0 && new_x_loc <= (scols - 1) && new_y_loc >= -1 && new_y_loc <= (srows - 2);
                    int mask3 =
                        new_x_loc >= -1 && new_x_loc <= (scols - 2) && new_y_loc >= -1 && new_y_loc <= (srows - 2);

                    int val_xy0 = 0;
                    int val_xy1 = 0;
                    int val_xy2 = 0;

                    if (mask0) {
                        val_xy0 += wtab[0] * src[loc_buffer];
                        val_xy1 += wtab[0] * src[loc_buffer + 1];
                        val_xy2 += wtab[0] * src[loc_buffer + 2];
                    }
                    if (mask1) {
                        val_xy0 += wtab[1] * src[loc_buffer + 3];
                        val_xy1 += wtab[1] * src[loc_buffer + 4];
                        val_xy2 += wtab[1] * src[loc_buffer + 5];
                    }
                    if (mask2) {
                        val_xy0 += wtab[2] * src2[loc_buffer];
                        val_xy1 += wtab[2] * src2[loc_buffer + 1];
                        val_xy2 += wtab[2] * src2[loc_buffer + 2];
                    }
                    if (mask3) {
                        val_xy0 += wtab[3] * src2[loc_buffer + 3];
                        val_xy1 += wtab[3] * src2[loc_buffer + 4];
                        val_xy2 += wtab[3] * src2[loc_buffer + 5];
                    }

                    dst[final_loc]     = SATURATE_CAST_UCHAR((val_xy0 + DELTA) >> 15);
                    dst[final_loc + 1] = SATURATE_CAST_UCHAR((val_xy1 + DELTA) >> 15);
                    dst[final_loc + 2] = SATURATE_CAST_UCHAR((val_xy2 + DELTA) >> 15);
                }
            }
        }

        int x = end_x - x_count + 1;

        for (; x <= end_x; x++) {
            int final_loc  = final_loc_base + x * 3;
            int loc_buffer = buf_loc[x];
            short* wtab = BilinearTab_i[tab_loc[x]][0];

            int point00 = src[loc_buffer];
            int point01 = src[loc_buffer + 1];
            int point02 = src[loc_buffer + 2];
            int point03 = src[loc_buffer + 3];
            int point04 = src[loc_buffer + 4];
            int point05 = src[loc_buffer + 5];
            int point10 = src2[loc_buffer];
            int point11 = src2[loc_buffer + 1];
            int point12 = src2[loc_buffer + 2];
            int point13 = src2[loc_buffer + 3];
            int point14 = src2[loc_buffer + 4];
            int point15 = src2[loc_buffer + 5];

            int val_xy0        = wtab[0] * point00 + wtab[1] * point03 + wtab[2] * point10 + wtab[3] * point13;
            int val_xy1        = wtab[0] * point01 + wtab[1] * point04 + wtab[2] * point11 + wtab[3] * point14;
            int val_xy2        = wtab[0] * point02 + wtab[1] * point05 + wtab[2] * point12 + wtab[3] * point15;
            dst[final_loc]     = SATURATE_CAST_UCHAR((val_xy0 + DELTA) >> 15);
            dst[final_loc + 1] = SATURATE_CAST_UCHAR((val_xy1 + DELTA) >> 15);
            dst[final_loc + 2] = SATURATE_CAST_UCHAR((val_xy2 + DELTA) >> 15);
        }
    }

    delete[] buf_loc;
    delete[] tab_loc;

    free(buffer);
}

}  // namespace TNN_NS
