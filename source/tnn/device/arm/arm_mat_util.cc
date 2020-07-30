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

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), 0), UCHAR_MAX)
#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X) (int)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

void mat_memcpy_2d(void* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    auto src_ptr = reinterpret_cast<uint8_t*>(src);
    auto dst_ptr = reinterpret_cast<uint8_t*>(dst);

    for (int h = 0; h < height; h++) {
        memcpy(dst_ptr, src_ptr, width);
        src_ptr += src_stride;
        dst_ptr += dst_stride;
    }

}

void resize_bilinear_c1(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return resize_bilinear_c1(src, src_w, src_h, src_w, dst, w, h, w);
}

void resize_bilinear_c2(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return resize_bilinear_c2(src, src_w, src_h, src_w * 2, dst, w, h, w * 2);
}

void resize_bilinear_c3(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return resize_bilinear_c3(src, src_w, src_h, src_w * 3, dst, w, h, w * 3);
}

void resize_bilinear_c4(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h) {
    return resize_bilinear_c4(src, src_w, src_h, src_w * 4, dst, w, h, w * 4);
}

void resize_bilinear_c1(const uint8_t* src, int src_w, int src_h, int src_stride, uint8_t* dst, int w, int h,
                        int stride) {
#if 0
    const int INTER_RESIZE_COEF_BITS  = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    //     const int ONE=INTER_RESIZE_COEF_SCALE;

    double scale_x = (double)src_w / w;
    double scale_y = (double)src_h / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;      // new int[w];
    int* yofs = buf + w;  // new int[h];

    short* ialpha = (short*)(buf + w + h);      // new short[w * 2];
    short* ibeta  = (short*)(buf + w + h + w);  // new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

// #define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++) {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0) {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= src_w - 1) {
            sx = src_w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 = fx * INTER_RESIZE_COEF_SCALE;

        ialpha[dx * 2]     = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++) {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0) {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= src_h - 1) {
            sy = src_h - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 = fy * INTER_RESIZE_COEF_SCALE;

        ibeta[dy * 2]     = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }

// #undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w, (size_t)2u);
    Mat rowsbuf1(w, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++) {
        int sy = yofs[dy];

        if (sy == prev_sy1) {
            // reuse all rows
        } else if (sy == prev_sy1 + 1) {
            // hresize one row
            short* rows0_old  = rows0;
            rows0             = rows1;
            rows1             = rows0_old;
            const uint8_t* S1 = src + src_stride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p        = rows1;
            for (int dx = 0; dx < w; dx++) {
                int sx   = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const uint8_t* S1p = S1 + sx;
                rows1p[dx]         = (S1p[0] * a0 + S1p[1] * a1) >> 4;

                ialphap += 2;
            }
        } else {
            // hresize two rows
            const uint8_t* S0 = src + src_stride * (sy);
            const uint8_t* S1 = src + src_stride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p        = rows0;
            short* rows1p        = rows1;
            for (int dx = 0; dx < w; dx++) {
                int sx   = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const uint8_t* S0p = S0 + sx;
                const uint8_t* S1p = S1 + sx;
                rows0p[dx]         = (S0p[0] * a0 + S0p[1] * a1) >> 4;
                rows1p[dx]         = (S1p[0] * a0 + S1p[1] * a1) >> 4;

                ialphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        uint8_t* Dp   = dst + stride * (dy);

#if __ARM_NEON
        int nn = w >> 3;
#else
        int nn = 0;
#endif
        int remain = w - (nn << 3);

#if __ARM_NEON
#if __aarch64__
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
#else
        if (nn > 0) {
            asm volatile(
                "vdup.s16   d16, %8         \n"
                "mov        r4, #2          \n"
                "vdup.s16   d17, %9         \n"
                "vdup.s32   q12, r4         \n"
                "pld        [%0, #128]      \n"
                "vld1.s16   {d2-d3}, [%0 :128]!\n"
                "pld        [%1, #128]      \n"
                "vld1.s16   {d6-d7}, [%1 :128]!\n"
                "0:                         \n"
                "vmull.s16  q0, d2, d16     \n"
                "vmull.s16  q1, d3, d16     \n"
                "vorr.s32   q10, q12, q12   \n"
                "vorr.s32   q11, q12, q12   \n"
                "vmull.s16  q2, d6, d17     \n"
                "vmull.s16  q3, d7, d17     \n"
                "vsra.s32   q10, q0, #16    \n"
                "vsra.s32   q11, q1, #16    \n"
                "pld        [%0, #128]      \n"
                "vld1.s16   {d2-d3}, [%0 :128]!\n"
                "vsra.s32   q10, q2, #16    \n"
                "vsra.s32   q11, q3, #16    \n"
                "pld        [%1, #128]      \n"
                "vld1.s16   {d6-d7}, [%1 :128]!\n"
                "vshrn.s32  d20, q10, #2    \n"
                "vshrn.s32  d21, q11, #2    \n"
                "vqmovun.s16 d20, q10        \n"
                "vst1.8     {d20}, [%2]!    \n"
                "subs       %3, #1          \n"
                "bne        0b              \n"
                "sub        %0, #16         \n"
                "sub        %1, #16         \n"
                : "=r"(rows0p),  // %0
                  "=r"(rows1p),  // %1
                  "=r"(Dp),      // %2
                  "=r"(nn)       // %3
                : "0"(rows0p), "1"(rows1p), "2"(Dp), "3"(nn),
                  "r"(b0),  // %8
                  "r"(b1)   // %9
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12");
        }
#endif  // __aarch64__
#endif  // __ARM_NEON
        for (; remain; --remain) {
            //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (uint8_t)(
                ((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }

        ibeta += 2;
    }

    delete[] buf;
#endif
}

void resize_bilinear_c2(const uint8_t* src, int src_w, int src_h, int src_stride, uint8_t* dst, int w, int h,
                        int stride) {
#if 0
    const int INTER_RESIZE_COEF_BITS  = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    //     const int ONE=INTER_RESIZE_COEF_SCALE;

    double scale_x = (double)src_w / w;
    double scale_y = (double)src_h / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;      // new int[w];
    int* yofs = buf + w;  // new int[h];

    short* ialpha = (short*)(buf + w + h);      // new short[w * 2];
    short* ibeta  = (short*)(buf + w + h + w);  // new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

// #define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++) {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0) {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= src_w - 1) {
            sx = src_w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx * 2;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 = fx * INTER_RESIZE_COEF_SCALE;

        ialpha[dx * 2]     = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++) {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0) {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= src_h - 1) {
            sy = src_h - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 = fy * INTER_RESIZE_COEF_SCALE;

        ibeta[dy * 2]     = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }

// #undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w * 2 + 2, (size_t)2u);
    Mat rowsbuf1(w * 2 + 2, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++) {
        int sy = yofs[dy];

        if (sy == prev_sy1) {
            // reuse all rows
        } else if (sy == prev_sy1 + 1) {
            // hresize one row
            short* rows0_old  = rows0;
            rows0             = rows1;
            rows1             = rows0_old;
            const uint8_t* S1 = src + src_stride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p        = rows1;
            for (int dx = 0; dx < w; dx++) {
                int sx = xofs[dx];

                const uint8_t* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0a1XX   = vld1_s16(ialphap);
                int16x4_t _a0a0a1a1 = vzip_s16(_a0a1XX, _a0a1XX).val[0];
                uint8x8_t _S1       = uint8x8_t();

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                _S1 = vld1_lane_u8(S1p + 3, _S1, 3);

                int16x8_t _S116      = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1lowhigh = vget_low_s16(_S116);
                int32x4_t _S1ma0a1   = vmull_s16(_S1lowhigh, _a0a0a1a1);
                int32x2_t _rows1low  = vadd_s32(vget_low_s32(_S1ma0a1), vget_high_s32(_S1ma0a1));
                int32x4_t _rows1     = vcombine_s32(_rows1low, vget_high_s32(_S1ma0a1));
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                rows1p[0] = (S1p[0] * a0 + S1p[2] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[3] * a1) >> 4;
#endif  // __ARM_NEON

                ialphap += 2;
                rows1p += 2;
            }
        } else {
            // hresize two rows
            const uint8_t* S0 = src + src_stride * (sy);
            const uint8_t* S1 = src + src_stride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p        = rows0;
            short* rows1p        = rows1;
            for (int dx = 0; dx < w; dx++) {
                int sx   = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const uint8_t* S0p = S0 + sx;
                const uint8_t* S1p = S1 + sx;
#if __ARM_NEON
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

                int16x8_t _S016      = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116      = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0lowhigh = vget_low_s16(_S016);
                int16x4_t _S1lowhigh = vget_low_s16(_S116);
                int32x2x2_t _S0S1low_S0S1high =
                    vtrn_s32(vreinterpret_s32_s16(_S0lowhigh), vreinterpret_s32_s16(_S1lowhigh));
                int32x4_t _rows01     = vmull_s16(vreinterpret_s16_s32(_S0S1low_S0S1high.val[0]), _a0);
                _rows01               = vmlal_s16(_rows01, vreinterpret_s16_s32(_S0S1low_S0S1high.val[1]), _a1);
                int16x4_t _rows01_sr4 = vshrn_n_s32(_rows01, 4);
                int16x4_t _rows1_sr4  = vext_s16(_rows01_sr4, _rows01_sr4, 2);
                vst1_s16(rows0p, _rows01_sr4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows0p[0] = (S0p[0] * a0 + S0p[2] * a1) >> 4;
                rows0p[1] = (S0p[1] * a0 + S0p[3] * a1) >> 4;
                rows1p[0] = (S1p[0] * a0 + S1p[2] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[3] * a1) >> 4;
#endif  // __ARM_NEON

                ialphap += 2;
                rows0p += 2;
                rows1p += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        uint8_t* Dp   = dst + stride * (dy);

#if __ARM_NEON
        int nn = (w * 2) >> 3;
#else
        int nn = 0;
#endif
        int remain = (w * 2) - (nn << 3);

#if __ARM_NEON
#if __aarch64__
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
#else
        if (nn > 0) {
            asm volatile(
                "vdup.s16   d16, %8         \n"
                "mov        r4, #2          \n"
                "vdup.s16   d17, %9         \n"
                "vdup.s32   q12, r4         \n"
                "pld        [%0, #128]      \n"
                "vld1.s16   {d2-d3}, [%0 :128]!\n"
                "pld        [%1, #128]      \n"
                "vld1.s16   {d6-d7}, [%1 :128]!\n"
                "0:                         \n"
                "vmull.s16  q0, d2, d16     \n"
                "vmull.s16  q1, d3, d16     \n"
                "vorr.s32   q10, q12, q12   \n"
                "vorr.s32   q11, q12, q12   \n"
                "vmull.s16  q2, d6, d17     \n"
                "vmull.s16  q3, d7, d17     \n"
                "vsra.s32   q10, q0, #16    \n"
                "vsra.s32   q11, q1, #16    \n"
                "pld        [%0, #128]      \n"
                "vld1.s16   {d2-d3}, [%0 :128]!\n"
                "vsra.s32   q10, q2, #16    \n"
                "vsra.s32   q11, q3, #16    \n"
                "pld        [%1, #128]      \n"
                "vld1.s16   {d6-d7}, [%1 :128]!\n"
                "vshrn.s32  d20, q10, #2    \n"
                "vshrn.s32  d21, q11, #2    \n"
                "vqmovun.s16 d20, q10        \n"
                "vst1.8     {d20}, [%2]!    \n"
                "subs       %3, #1          \n"
                "bne        0b              \n"
                "sub        %0, #16         \n"
                "sub        %1, #16         \n"
                : "=r"(rows0p),  // %0
                  "=r"(rows1p),  // %1
                  "=r"(Dp),      // %2
                  "=r"(nn)       // %3
                : "0"(rows0p), "1"(rows1p), "2"(Dp), "3"(nn),
                  "r"(b0),  // %8
                  "r"(b1)   // %9
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12");
        }
#endif  // __aarch64__
#endif  // __ARM_NEON
        for (; remain; --remain) {
            //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (uint8_t)(
                ((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }

        ibeta += 2;
    }

    delete[] buf;
#endif
}

void resize_bilinear_c3(const uint8_t* src, int src_w, int src_h, int src_stride, uint8_t* dst, int w, int h,
                        int stride) {
#if 0
    const int INTER_RESIZE_COEF_BITS  = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    //     const int ONE=INTER_RESIZE_COEF_SCALE;

    double scale_x = (double)src_w / w;
    double scale_y = (double)src_h / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;      // new int[w];
    int* yofs = buf + w;  // new int[h];

    short* ialpha = (short*)(buf + w + h);      // new short[w * 2];
    short* ibeta  = (short*)(buf + w + h + w);  // new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

// #define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++) {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0) {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= src_w - 1) {
            sx = src_w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx * 3;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 = fx * INTER_RESIZE_COEF_SCALE;

        ialpha[dx * 2]     = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++) {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0) {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= src_h - 1) {
            sy = src_h - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 = fy * INTER_RESIZE_COEF_SCALE;

        ibeta[dy * 2]     = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }

// #undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w * 3 + 1, (size_t)2u);
    Mat rowsbuf1(w * 3 + 1, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++) {
        int sy = yofs[dy];

        if (sy == prev_sy1) {
            // reuse all rows
        } else if (sy == prev_sy1 + 1) {
            // hresize one row
            short* rows0_old  = rows0;
            rows0             = rows1;
            rows1             = rows0_old;
            const uint8_t* S1 = src + src_stride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p        = rows1;
            for (int dx = 0; dx < w; dx++) {
                int sx   = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const uint8_t* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S1 = uint8x8_t();

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                _S1 = vld1_lane_u8(S1p + 3, _S1, 3);
                _S1 = vld1_lane_u8(S1p + 4, _S1, 4);
                _S1 = vld1_lane_u8(S1p + 5, _S1, 5);

                int16x8_t _S116      = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1low     = vget_low_s16(_S116);
                int16x4_t _S1high    = vext_s16(_S1low, vget_high_s16(_S116), 3);
                int32x4_t _rows1     = vmull_s16(_S1low, _a0);
                _rows1               = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows1p[0] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;
#endif  // __ARM_NEON

                ialphap += 2;
                rows1p += 3;
            }
        } else {
            // hresize two rows
            const uint8_t* S0 = src + src_stride * (sy);
            const uint8_t* S1 = src + src_stride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p        = rows0;
            short* rows1p        = rows1;
            for (int dx = 0; dx < w; dx++) {
                int sx   = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const uint8_t* S0p = S0 + sx;
                const uint8_t* S1p = S1 + sx;
#if __ARM_NEON
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

                int16x8_t _S016      = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116      = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0low     = vget_low_s16(_S016);
                int16x4_t _S1low     = vget_low_s16(_S116);
                int16x4_t _S0high    = vext_s16(_S0low, vget_high_s16(_S016), 3);
                int16x4_t _S1high    = vext_s16(_S1low, vget_high_s16(_S116), 3);
                int32x4_t _rows0     = vmull_s16(_S0low, _a0);
                int32x4_t _rows1     = vmull_s16(_S1low, _a0);
                _rows0               = vmlal_s16(_rows0, _S0high, _a1);
                _rows1               = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows0p, _rows0_sr4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows0p[0] = (S0p[0] * a0 + S0p[3] * a1) >> 4;
                rows0p[1] = (S0p[1] * a0 + S0p[4] * a1) >> 4;
                rows0p[2] = (S0p[2] * a0 + S0p[5] * a1) >> 4;
                rows1p[0] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;
#endif  // __ARM_NEON

                ialphap += 2;
                rows0p += 3;
                rows1p += 3;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        uint8_t* Dp   = dst + stride * (dy);

#if __ARM_NEON
        int nn = (w * 3) >> 3;
#else
        int nn = 0;
#endif
        int remain = (w * 3) - (nn << 3);

#if __ARM_NEON
#if __aarch64__
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
#else
        if (nn > 0) {
            asm volatile(
                "vdup.s16   d16, %8         \n"
                "mov        r4, #2          \n"
                "vdup.s16   d17, %9         \n"
                "vdup.s32   q12, r4         \n"
                "pld        [%0, #128]      \n"
                "vld1.s16   {d2-d3}, [%0 :128]!\n"
                "pld        [%1, #128]      \n"
                "vld1.s16   {d6-d7}, [%1 :128]!\n"
                "0:                         \n"
                "vmull.s16  q0, d2, d16     \n"
                "vmull.s16  q1, d3, d16     \n"
                "vorr.s32   q10, q12, q12   \n"
                "vorr.s32   q11, q12, q12   \n"
                "vmull.s16  q2, d6, d17     \n"
                "vmull.s16  q3, d7, d17     \n"
                "vsra.s32   q10, q0, #16    \n"
                "vsra.s32   q11, q1, #16    \n"
                "pld        [%0, #128]      \n"
                "vld1.s16   {d2-d3}, [%0 :128]!\n"
                "vsra.s32   q10, q2, #16    \n"
                "vsra.s32   q11, q3, #16    \n"
                "pld        [%1, #128]      \n"
                "vld1.s16   {d6-d7}, [%1 :128]!\n"
                "vshrn.s32  d20, q10, #2    \n"
                "vshrn.s32  d21, q11, #2    \n"
                "vqmovun.s16 d20, q10        \n"
                "vst1.8     {d20}, [%2]!    \n"
                "subs       %3, #1          \n"
                "bne        0b              \n"
                "sub        %0, #16         \n"
                "sub        %1, #16         \n"
                : "=r"(rows0p),  // %0
                  "=r"(rows1p),  // %1
                  "=r"(Dp),      // %2
                  "=r"(nn)       // %3
                : "0"(rows0p), "1"(rows1p), "2"(Dp), "3"(nn),
                  "r"(b0),  // %8
                  "r"(b1)   // %9
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12");
        }
#endif  // __aarch64__
#endif  // __ARM_NEON
        for (; remain; --remain) {
            //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (uint8_t)(
                ((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }

        ibeta += 2;
    }

    delete[] buf;
#endif
}

void resize_bilinear_c4(const uint8_t* src, int src_w, int src_h, int src_stride, uint8_t* dst, int w, int h,
                        int stride) {
#if 0
    const int INTER_RESIZE_COEF_BITS  = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    //     const int ONE=INTER_RESIZE_COEF_SCALE;

    double scale_x = (double)src_w / w;
    double scale_y = (double)src_h / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;      // new int[w];
    int* yofs = buf + w;  // new int[h];

    short* ialpha = (short*)(buf + w + h);      // new short[w * 2];
    short* ibeta  = (short*)(buf + w + h + w);  // new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

// #define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++) {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0) {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= src_w - 1) {
            sx = src_w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx * 4;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 = fx * INTER_RESIZE_COEF_SCALE;

        ialpha[dx * 2]     = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++) {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0) {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= src_h - 1) {
            sy = src_h - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 = fy * INTER_RESIZE_COEF_SCALE;

        ibeta[dy * 2]     = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }

// #undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w * 4, (size_t)2u);
    Mat rowsbuf1(w * 4, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++) {
        int sy = yofs[dy];

        if (sy == prev_sy1) {
            // reuse all rows
        } else if (sy == prev_sy1 + 4) {
            // hresize one row
            short* rows0_old  = rows0;
            rows0             = rows1;
            rows1             = rows0_old;
            const uint8_t* S1 = src + src_stride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p        = rows1;
            for (int dx = 0; dx < w; dx++) {
                int sx   = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const uint8_t* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0        = vdup_n_s16(a0);
                int16x4_t _a1        = vdup_n_s16(a1);
                uint8x8_t _S1        = vld1_u8(S1p);
                int16x8_t _S116      = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1low     = vget_low_s16(_S116);
                int16x4_t _S1high    = vget_high_s16(_S116);
                int32x4_t _rows1     = vmull_s16(_S1low, _a0);
                _rows1               = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows1p[0] = (S1p[0] * a0 + S1p[4] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[5] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[6] * a1) >> 4;
                rows1p[3] = (S1p[3] * a0 + S1p[7] * a1) >> 4;
#endif  // __ARM_NEON

                ialphap += 2;
                rows1p += 4;
            }
        } else {
            // hresize two rows
            const uint8_t* S0 = src + src_stride * (sy);
            const uint8_t* S1 = src + src_stride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p        = rows0;
            short* rows1p        = rows1;
            for (int dx = 0; dx < w; dx++) {
                int sx   = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const uint8_t* S0p = S0 + sx;
                const uint8_t* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0        = vdup_n_s16(a0);
                int16x4_t _a1        = vdup_n_s16(a1);
                uint8x8_t _S0        = vld1_u8(S0p);
                uint8x8_t _S1        = vld1_u8(S1p);
                int16x8_t _S016      = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116      = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0low     = vget_low_s16(_S016);
                int16x4_t _S1low     = vget_low_s16(_S116);
                int16x4_t _S0high    = vget_high_s16(_S016);
                int16x4_t _S1high    = vget_high_s16(_S116);
                int32x4_t _rows0     = vmull_s16(_S0low, _a0);
                int32x4_t _rows1     = vmull_s16(_S1low, _a0);
                _rows0               = vmlal_s16(_rows0, _S0high, _a1);
                _rows1               = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows0p, _rows0_sr4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows0p[0] = (S0p[0] * a0 + S0p[4] * a1) >> 4;
                rows0p[1] = (S0p[1] * a0 + S0p[5] * a1) >> 4;
                rows0p[2] = (S0p[2] * a0 + S0p[6] * a1) >> 4;
                rows0p[3] = (S0p[3] * a0 + S0p[7] * a1) >> 4;
                rows1p[0] = (S1p[0] * a0 + S1p[4] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[5] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[6] * a1) >> 4;
                rows1p[3] = (S1p[3] * a0 + S1p[7] * a1) >> 4;
#endif  // __ARM_NEON

                ialphap += 2;
                rows0p += 4;
                rows1p += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        uint8_t* Dp   = dst + stride * (dy);

#if __ARM_NEON
        int nn = (w * 4) >> 3;
#else
        int nn = 0;
#endif
        int remain = (w * 4) - (nn << 3);

#if __ARM_NEON
#if __aarch64__
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
#else
        if (nn > 0) {
            asm volatile(
                "vdup.s16   d16, %8         \n"
                "mov        r4, #2          \n"
                "vdup.s16   d17, %9         \n"
                "vdup.s32   q12, r4         \n"
                "pld        [%0, #128]      \n"
                "vld1.s16   {d2-d3}, [%0 :128]!\n"
                "pld        [%1, #128]      \n"
                "vld1.s16   {d6-d7}, [%1 :128]!\n"
                "0:                         \n"
                "vmull.s16  q0, d2, d16     \n"
                "vmull.s16  q1, d3, d16     \n"
                "vorr.s32   q10, q12, q12   \n"
                "vorr.s32   q11, q12, q12   \n"
                "vmull.s16  q2, d6, d17     \n"
                "vmull.s16  q3, d7, d17     \n"
                "vsra.s32   q10, q0, #16    \n"
                "vsra.s32   q11, q1, #16    \n"
                "pld        [%0, #128]      \n"
                "vld1.s16   {d2-d3}, [%0 :128]!\n"
                "vsra.s32   q10, q2, #16    \n"
                "vsra.s32   q11, q3, #16    \n"
                "pld        [%1, #128]      \n"
                "vld1.s16   {d6-d7}, [%1 :128]!\n"
                "vshrn.s32  d20, q10, #2    \n"
                "vshrn.s32  d21, q11, #2    \n"
                "vqmovun.s16 d20, q10        \n"
                "vst1.8     {d20}, [%2]!    \n"
                "subs       %3, #1          \n"
                "bne        0b              \n"
                "sub        %0, #16         \n"
                "sub        %1, #16         \n"
                : "=r"(rows0p),  // %0
                  "=r"(rows1p),  // %1
                  "=r"(Dp),      // %2
                  "=r"(nn)       // %3
                : "0"(rows0p), "1"(rows1p), "2"(Dp), "3"(nn),
                  "r"(b0),  // %8
                  "r"(b1)   // %9
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12");
        }
#endif  // __aarch64__
#endif  // __ARM_NEON
        for (; remain; --remain) {
            //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (uint8_t)(
                ((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }

        ibeta += 2;
    }

    delete[] buf;
#endif
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

#if 0
static short BilinearTab_i[1024][2][2];
#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << 15)

static inline void interpolateLinear(float x, float* coeffs) {
    coeffs[0] = 1.f - x;
    coeffs[1] = x;
}

static void initInterTab1D(float* tab, int tabsz) {
    float scale = 1.f / tabsz;
    for (int i = 0; i < tabsz; i++, tab += 2)
        interpolateLinear(i * scale, tab);
}

static void initInterTab2D() {
    short* itab = 0;
    int ksize   = 0;
    itab = BilinearTab_i[0][0], ksize = 2;

    float* _tab = new float[8 * cv::INTER_TAB_SIZE];
    int i, j, k1, k2;
    initInterTab1D(_tab, cv::INTER_TAB_SIZE);
    for (i = 0; i < cv::INTER_TAB_SIZE; i++) {
        for (j = 0; j < cv::INTER_TAB_SIZE; j++, itab += ksize * ksize) {
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

#define AB_SCALE (1 << 10)
#define round_delta 16
#define SIMD_LENGTH 2
#endif

void warpaffine_bilinear_c1(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h,
                            const float (*transform)[3]) {
#if 0
    int dst_w = w;
    int dst_h = h;

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

    if (!(flags & cv::WARP_INVERSE_MAP)) {
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
    } else {
        m[0] = M[0];
        m[1] = M[1];
        m[2] = M[2];
        m[3] = M[3];
        m[4] = M[4];
        m[5] = M[5];
    }

    int* buffer;
    posix_memalign(reinterpret_cast<void**>(&buffer), 32, (dst_w + dst_h) * 2 * sizeof(int));

    int* adelta = buffer;
    int* bdelta = buffer + dst_w * 2;

    int* ptra = adelta;
    int* ptrb = bdelta;
    for (int x = 0; x < dst_w; x++) {
        *ptra++ = SATURATE_CAST_INT(m[0] * x * 1024);
        *ptra++ = SATURATE_CAST_INT(m[3] * x * 1024);
    }

    for (int y = 0; y < dst_h; y++) {
        *ptrb++ = SATURATE_CAST_INT((m[1] * y + m[2]) * 1024);
        *ptrb++ = SATURATE_CAST_INT((m[4] * y + m[5]) * 1024);
    }

    int DELTA = 1 << 14;

    int scols             = src_w;
    int srows             = src_h;
    int schannel          = 1;
    int stmp              = scols * schannel;
    unsigned int* buf_loc = new unsigned int[dst_w];
    short* tab_loc        = new short[dst_w];

    unsigned short* buf_point = (unsigned short*)buf_loc;
    uchar* src2               = src + stmp;

    for (int y = 0; y < dst_h; ++y) {
        int x_count        = 0;
        int end_x          = 0;
        int final_loc_base = y * dst_w;
        for (int x = 0; x < dst_w; ++x) {
            int final_loc = final_loc_base + x;
            // int new_x = adelta[x] + offsets[2 * y] + 16;
            // int new_y = bdelta[x] + offsets[2 * y + 1] + 16;
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
        uchar* ptr = (uchar*)(buf_loc + x);

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

        int32x4_t DELTA_vec = vdupq_n_s32(DELTA);
        uchar* dst_loc      = dst + final_loc_base + x;

#if __aarch64__
        short* BilinearTab_ptr = BilinearTab_i[0][0];
        int cmp_flag           = end_x - 8 + 1;
        int simd_loop          = x_count >> 3;
        if (simd_loop > 0) {
            asm volatile(
                "subs x12, %7, #1\n\t"
                "blt 1f\n\t"
                "#load from tab_loc\n\t"
                "add x7, %4, %5, lsl #1\n\t"
                "ldrsh x8, [x7]\n\t"
                "ldrsh x9, [x7, #2]\n\t"
                "ldrsh x10, [x7, #4]\n\t"
                "ldrsh x11, [x7, #6]\n\t"
                "#load from ptr\n\t"
                "ld1 {v0.4s}, [%0], #16\n\t"
                "add x8, %3, x8, lsl #3\n\t"
                "add x9, %3, x9, lsl #3\n\t"
                "add x10, %3, x10, lsl #3\n\t"
                "add x11, %3, x11, lsl #3\n\t"
                "0:\n\t"
                "ld1 {v1.4s}, [%0], #16\n\t"
                "ins v2.s[0], v0.s[1]\n\t"
                "ins v3.s[0], v0.s[2]\n\t"
                "ins v4.s[0], v0.s[3]\n\t"
                "#load from BilinearTab\n\t"
                "ld1 {v8.4h}, [x8]\n\t"
                "ld1 {v9.4h}, [x9]\n\t"
                "ld1 {v10.4h}, [x10]\n\t"
                "ld1 {v11.4h}, [x11]\n\t"
                "ldrsh x8, [x7, #8]\n\t"
                "ldrsh x9, [x7, #10]\n\t"
                "ldrsh x10, [x7, #12]\n\t"
                "ldrsh x11, [x7, #14]\n\t"
                "add x7, x7, #16\n\t"
                "#start calculation\n\t"
                "ushll v0.8h, v0.8b, #0\n\t"
                "ushll v2.8h, v2.8b, #0\n\t"
                "ushll v3.8h, v3.8b, #0\n\t"
                "ushll v4.8h, v4.8b, #0\n\t"
                "ins v5.s[0], v1.s[1]\n\t"
                "ins v6.s[0], v1.s[2]\n\t"
                "ins v7.s[0], v1.s[3]\n\t"
                "add x8, %3, x8, lsl #3\n\t"
                "add x9, %3, x9, lsl #3\n\t"
                "add x10, %3, x10, lsl #3\n\t"
                "add x11, %3, x11, lsl #3\n\t"
                "smull v0.4s, v0.4h, v8.4h\n\t"
                "smull v2.4s, v2.4h, v9.4h\n\t"
                "smull v3.4s, v3.4h, v10.4h\n\t"
                "smull v4.4s, v4.4h, v11.4h\n\t"
                "ld1 {v8.4h}, [x8]\n\t"
                "ld1 {v9.4h}, [x9]\n\t"
                "ld1 {v10.4h}, [x10]\n\t"
                "ld1 {v11.4h}, [x11]\n\t"
                "ushll v1.8h, v1.8b, #0\n\t"
                "ushll v5.8h, v5.8b, #0\n\t"
                "ushll v6.8h, v6.8b, #0\n\t"
                "ushll v7.8h, v7.8b, #0\n\t"
                "addp v0.4s, v0.4s, v2.4s\n\t"
                "addp v3.4s, v3.4s, v4.4s\n\t"
                "smull v1.4s, v1.4h, v8.4h\n\t"
                "smull v5.4s, v5.4h, v9.4h\n\t"
                "smull v6.4s, v6.4h, v10.4h\n\t"
                "smull v7.4s, v7.4h, v11.4h\n\t"
                "addp v0.4s, v0.4s, v3.4s\n\t"
                "ldrsh x8, [x7]\n\t"
                "ldrsh x9, [x7, #2]\n\t"
                "ldrsh x10, [x7, #4]\n\t"
                "ldrsh x11, [x7, #6]\n\t"
                "addp v1.4s, v1.4s, v5.4s\n\t"
                "addp v6.4s, v6.4s, v7.4s\n\t"
                "add v2.4s, v0.4s, %6.4s\n\t"
                "ld1 {v0.4s}, [%0], #16\n\t"
                "add x8, %3, x8, lsl #3\n\t"
                "add x9, %3, x9, lsl #3\n\t"
                "add x10, %3, x10, lsl #3\n\t"
                "add x11, %3, x11, lsl #3\n\t"
                "addp v1.4s, v1.4s, v6.4s\n\t"
                "shrn v2.4h, v2.4s, #15\n\t"
                "add v1.4s, v1.4s, %6.4s\n\t"
                "subs x12, x12, #1\n\t"
                "shrn v1.4h, v1.4s, #15\n\t"
                "ins v2.d[1], v1.d[0]\n\t"
                "sqxtun v2.8b, v2.8h\n\t"
                "st1 {v2.8b}, [%2], #8\n\t"
                "bge 0b\n\t"
                "sub %0, %0, #16\n\t"
                "1:\n\t"
                : "=r"(ptr)
                : "0"(ptr), "r"(dst_loc), "r"(BilinearTab_ptr), "r"(tab_loc), "r"(x), "w"(DELTA_vec), "r"(simd_loop)
                : "cc", "memory", "x7", "x8", "x9", "x10", "x11", "x12", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                  "v8", "v9", "v10", "v11");
            x = x + (simd_loop << 3);
        }
#else
        short* BilinearTab_ptr = BilinearTab_i[0][0];
        for (; x <= end_x - 8 + 1; x += 8) {
            asm volatile(
                "#a32 mark1\n\t"
                "#load from tab_loc\n\t"
                "add r7, %5, %6, lsl #1\n\t"
                "vld1.16 {d8}, [r7]!\n\t"
                "#load from ptr\n\t"
                "vld1.32 {d0[0]}, [%0]!\n\t"
                "#load from BilinearTab\n\t"
                "vmov.s16 r8, d8[0]\n\t"
                "vld1.32 {d2[0]}, [%0]!\n\t"
                "vmov.s16 r9, d8[1]\n\t"
                "vld1.32 {d4[0]}, [%0]!\n\t"
                "vmov.s16 r10, d8[2]\n\t"
                "vld1.32 {d6[0]}, [%0]!\n\t"
                // "vmov.s16 r11, d8[3]\n\t"
                "vld1.16 {d12}, [r7]\n\t"
                "vmov.s16 r7, d8[3]\n"
                "add r8, %4, r8, lsl #3\n\t"
                "add r9, %4, r9, lsl #3\n\t"
                "add r10, %4, r10, lsl #3\n\t"
                // "add r11, %4, r11, lsl #3\n\t"
                "add r7, %4, r7, lsl #3\n\t"
                "vld1.16 {d8}, [r8]\n\t"
                "vld1.16 {d9}, [r9]\n\t"
                "vld1.16 {d10}, [r10]\n\t"
                "vld1.16 {d11}, [r7]\n\t"
                "vmov.s16 r8, d12[0]\n\t"
                "vmov.s16 r9, d12[1]\n\t"
                "vmov.s16 r10, d12[2]\n\t"
                "vmov.s16 r7, d12[3]\n\t"

                "vld1.32 {d12[0]}, [%0]!\n\t"
                "add r8, %4, r8, lsl #3\n\t"
                "add r9, %4, r9, lsl #3\n\t"
                "add r10, %4, r10, lsl #3\n\t"
                "add r7, %4, r7, lsl #3\n\t"
                "vld1.32 {d14[0]}, [%0]!\n\t"

                "#calculate p0*w0\n\t"
                "vmovl.u8 q0, d0\n\t"
                "vld1.32 {d16[0]}, [%0]!\n\t"
                "vmovl.u8 q1, d2\n\t"
                "vmovl.u8 q2, d4\n\t"
                "vmovl.u8 q3, d6\n\t"
                "vld1.32 {d18[0]}, [%0]!\n\t"

                "vld1.16 {d20}, [r8]\n\t"
                "vld1.16 {d21}, [r9]\n\t"
                "vld1.16 {d22}, [r10]\n\t"
                "vld1.16 {d23}, [r7]\n\t"

                "vmull.s16 q0, d0, d8\n\t"
                "vmull.s16 q1, d2, d9\n\t"
                "vmull.s16 q2, d4, d10\n\t"
                "vmull.s16 q3, d6, d11\n\t"

                "vpadd.i32 d0, d0, d1\n\t"
                "vpadd.i32 d1, d2, d3\n\t"
                "vpadd.i32 d2, d4, d5\n\t"
                "vpadd.i32 d3, d6, d7\n\t"

                "vmovl.u8 q6, d12\n\t"
                "vmovl.u8 q7, d14\n\t"
                "vmovl.u8 q8, d16\n\t"
                "vmovl.u8 q9, d18\n\t"

                "vpadd.i32 d0, d0, d1\n\t"
                "vpadd.i32 d1, d2, d3\n\t"

                "vmull.s16 q6, d12, d20\n\t"
                "vmull.s16 q7, d14, d21\n\t"
                "vmull.s16 q8, d16, d22\n\t"
                "vmull.s16 q9, d18, d23\n\t"

                "vadd.i32 q0, q0, %q7\n\t"
                "vshrn.i32 d0, q0, #15\n\t"

                "vpadd.i32 d12, d12, d13\n\t"
                "vpadd.i32 d13, d14, d15\n\t"
                "vpadd.i32 d14, d16, d17\n\t"
                "vpadd.i32 d15, d18, d19\n\t"

                "vpadd.i32 d12, d12, d13\n\t"
                "vpadd.i32 d13, d14, d15\n\t"

                "vadd.i32 q6, q6, %q7\n\t"
                "vshrn.i32 d1, q6, #15\n\t"

                "vmovn.i16 d0, q0\n\t"
                "vst1.8 {d0}, [%1]!\n\t"

                : "=r"(ptr), "=r"(dst_loc)
                : "0"(ptr), "1"(dst_loc), "r"(BilinearTab_ptr), "r"(tab_loc), "r"(x), "w"(DELTA_vec)
                : "cc", "memory", "r7", "r8", "r9", "r10", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                  "q10", "q11");
        }
#endif  // a64
#endif  // end of NEON

        for (; x <= end_x; x++) {
            int final_loc = final_loc_base + x;
            short* wtab   = BilinearTab_i[tab_loc[x]][0];

            // uchar * ptr = (uchar*)(buf_loc + x);
            int point0 = ptr[0];
            int point1 = ptr[1];
            int point2 = ptr[2];
            int point3 = ptr[3];
            ptr += 4;

            // int point0 = src[loc_buffer];
            // int point1 = src[loc_buffer + 1];
            // int point2 = src[loc_buffer2];
            // int point3 = src[loc_buffer2 + 1];

            int val_xy0    = wtab[0] * point0 + wtab[1] * point1 + wtab[2] * point2 + wtab[3] * point3;
            dst[final_loc] = SATURATE_CAST_UCHAR((val_xy0 + DELTA) >> 15);
        }
    }
    delete[] buf_loc;
    delete[] tab_loc;

    free(buffer);
#endif
}

void warpaffine_bilinear_c3(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h,
                            const float (*transform)[3]) {
#if 0
    int dst_w = w;
    int dst_h = h;

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

    if (!(flags & cv::WARP_INVERSE_MAP)) {
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
    } else {
        m[0] = M[0];
        m[1] = M[1];
        m[2] = M[2];
        m[3] = M[3];
        m[4] = M[4];
        m[5] = M[5];
    }

    int* buffer;
    posix_memalign(reinterpret_cast<void**>(&buffer), 32, (dst_w + dst_h) * 2 * sizeof(int));

    int* adelta = buffer;
    int* bdelta = buffer + dst_w * 2;

    int* ptra = adelta;
    int* ptrb = bdelta;
    for (int x = 0; x < dst_w; x++) {
        *ptra++ = SATURATE_CAST_INT(m[0] * x * 1024);
        *ptra++ = SATURATE_CAST_INT(m[3] * x * 1024);
    }

    for (int y = 0; y < dst_h; y++) {
        *ptrb++ = SATURATE_CAST_INT((m[1] * y + m[2]) * 1024);
        *ptrb++ = SATURATE_CAST_INT((m[4] * y + m[5]) * 1024);
    }

    int DELTA = 1 << 14;

    int scols      = srcMat.cols;
    int srows      = srcMat.rows;
    int schannel   = srcMat.channels();
    int stmp       = scols * schannel;
    int* buf_loc   = new int[dst_w + 4];
    short* tab_loc = new short[dst_w + 4];

    uchar* src2 = src + stmp;

    short xy_loc_buf[dst_w * 2];
    short xy_float_buf[dst_w];

    for (int y = 0; y < dst_h; ++y) {
        int x_count        = 0;
        int end_x          = 0;
        int final_loc_base = y * dst_w * 3;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        int32x4_t off_vec    = vdupq_n_s32(16);
        int16x4_t mask31     = vdup_n_s16(31);
        int16x8_t mask_mull  = {1, 32, 1, 32, 1, 32, 1, 32};
        int32x4_t bdelta_vec = {bdelta[2 * y], bdelta[2 * y + 1], bdelta[2 * y], bdelta[2 * y + 1]};
        int idx              = 0;
        for (; idx < dst_w - 4 + 1; idx += 4) {
            int32x4_t adelta0 = vaddq_s32(vld1q_s32(adelta + 2 * idx), off_vec);
            int32x4_t adelta1 = vaddq_s32(vld1q_s32(adelta + 2 * idx + 4), off_vec);
            // x0y0,x1y1
            int32x4_t x0y0 = vaddq_s32(adelta0, bdelta_vec);
            // x2y2,x3y3
            int32x4_t x2y2     = vaddq_s32(adelta1, bdelta_vec);
            int16x4_t x0y0sh   = vshrn_n_s32(x0y0, 5);
            int16x4_t x2y2sh   = vshrn_n_s32(x2y2, 5);
            int16x8_t xy_float = vcombine_s16(vand_s16(x0y0sh, mask31), vand_s16(x2y2sh, mask31));
            xy_float           = vmulq_s16(xy_float, mask_mull);
            int16x8_t xy       = vcombine_s16(vshrn_n_s32(x0y0, 10), vshrn_n_s32(x2y2, 10));
            // xy_float = vpaddq_s16(xy_float, xy_float);
            int16x4_t xy_float0 = vpadd_s16(vget_low_s16(xy_float), vget_high_s16(xy_float));
            vst1q_s16(xy_loc_buf + idx * 2, xy);
            vst1_s16(xy_float_buf + idx, xy_float0);
            // vst1q_lane_s64((int64_t*)(xy_float_buf + idx), vreinterpretq_s64_s16(xy_float), 0);
        }
        for (; idx < dst_w; idx++) {
            int new_x               = adelta[2 * idx] + bdelta[2 * y] + 16;
            int new_y               = adelta[2 * idx + 1] + bdelta[2 * y + 1] + 16;
            int new_x_full          = new_x >> 5;
            int new_y_full          = new_y >> 5;
            xy_loc_buf[idx * 2]     = (new_x >> 10);
            xy_loc_buf[idx * 2 + 1] = (new_y >> 10);
            xy_float_buf[idx]       = (new_x_full & 31) + (new_y_full & 31) * 32;
        }

        for (int x = 0; x < dst_w; ++x) {
            // int new_x = adelta[2 * x] + bdelta[2 * y] + 16;
            // int new_y = adelta[2 * x + 1] + bdelta[2 * y + 1] + 16;
            // int new_x_full = new_x >> 5;
            // int new_y_full = new_y >> 5;
            // int new_x_loc = new_x >> 10;
            // int new_y_loc = new_y >> 10;
            //
            // short new_xy_float = (new_x_full & 31) + (new_y_full & 31) * 32;
            // short *wtab = BilinearTab_i[new_xy_float][0];
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
#endif

        int x = end_x - x_count + 1;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

        int32x4_t DELTA_vec = vdupq_n_s32(DELTA);
        uint8x8_t tb        = {0, 1, 2, 4, 5, 6, 0, 0};
        uchar* dst_loc      = dst + final_loc_base + x * 3;

#if __aarch64__
        short* BilinearTab_ptr = BilinearTab_i[0][0];
        int simd_loop          = x_count >> 2;
        int cmp_flag           = end_x - 4 + 1;
        if (simd_loop > 0) {
            asm volatile(
                "subs x25, %2, #1\n\t"
                "blt 1f\n\t"
                "add x17, %4, %3, lsl #1\n\t"
                "add x18, %5, %3, lsl #2\n\t"
                "ldrsh x19, [x17]\n\t"
                "ldrsh x20, [x17, #2]\n\t"
                "add x17, x17, #4\n\t"
                "ldpsw x21, x22, [x18], #8\n\t"
                "add x19, %6, x19, lsl #3\n\t"
                "add x20, %6, x20, lsl #3\n\t"
                "0:\n\t"
                "ldrsh x23, [x17]\n\t"
                "ldrsh x24, [x17, #2]\n\t"
                "add x17, x17, #4\n\t"
                // vec00 vec01 vec10 vec11
                "ldr d2, [%7, x21]\n\t"
                "ldr d4, [%7, x22]\n\t"
                "ldr d3, [%8, x21]\n\t"
                "ldr d5, [%8, x22]\n\t"
                // wtab0 and wtab1
                "ld1 {v0.4h}, [x19]\n\t"
                "ld1 {v1.4h}, [x20]\n\t"
                "ldpsw x21, x22, [x18], #8\n\t"
                "add x23, %6, x23, lsl #3\n\t"
                "add x24, %6, x24, lsl #3\n\t"
                // calculation of vec00,01,10,11
                "ushll v2.8h, v2.8b, #0\n\t"
                "ushll v3.8h, v3.8b, #0\n\t"
                "ushll v4.8h, v4.8b, #0\n\t"
                "ushll v5.8h, v5.8b, #0\n\t"
                "mov v6.d[0], v2.d[1]\n\t"
                "mov v7.d[0], v3.d[1]\n\t"
                "mov v8.d[0], v4.d[1]\n\t"
                "mov v9.d[0], v5.d[1]\n\t"
                // vec20 vec21 vec30 vec31
                "ldr d16, [%7, x21]\n\t"
                "ldr d18, [%7, x22]\n\t"
                "ldr d17, [%8, x21]\n\t"
                "ldr d19, [%8, x22]\n\t"
                "ext v6.8b, v2.8b, v6.8b, #6\n\t"
                "ext v7.8b, v3.8b, v7.8b, #6\n\t"
                "ext v8.8b, v4.8b, v8.8b, #6\n\t"
                "ext v9.8b, v5.8b, v9.8b, #6\n\t"
                // wtab2 and wtab3
                "ld1 {v14.4h}, [x23]\n\t"
                "ld1 {v15.4h}, [x24]\n\t"
                "smull v10.4s, v2.4h, v0.h[0]\n\t"
                "smull v11.4s, v3.4h, v0.h[2]\n\t"
                "smull v12.4s, v4.4h, v1.h[0]\n\t"
                "smull v13.4s, v5.4h, v1.h[2]\n\t"
                "smlal v10.4s, v6.4h, v0.h[1]\n\t"
                "smlal v11.4s, v7.4h, v0.h[3]\n\t"
                "smlal v12.4s, v8.4h, v1.h[1]\n\t"
                "smlal v13.4s, v9.4h, v1.h[3]\n\t"
                // next loop
                "ldrsh x19, [x17]\n\t"
                "ldrsh x20, [x17, #2]\n\t"
                "add x17, x17, #4\n\t"
                "ldpsw x21, x22, [x18], #8\n\t"
                // calculation of vec00,01,10,11
                "ushll v16.8h, v16.8b, #0\n\t"
                "ushll v17.8h, v17.8b, #0\n\t"
                "ushll v18.8h, v18.8b, #0\n\t"
                "ushll v19.8h, v19.8b, #0\n\t"
                "add x19, %6, x19, lsl #3\n\t"
                "add x20, %6, x20, lsl #3\n\t"
                "mov v6.d[0], v16.d[1]\n\t"
                "mov v7.d[0], v17.d[1]\n\t"
                "mov v8.d[0], v18.d[1]\n\t"
                "mov v9.d[0], v19.d[1]\n\t"
                "ext v6.8b, v16.8b, v6.8b, #6\n\t"
                "ext v7.8b, v17.8b, v7.8b, #6\n\t"
                "ext v8.8b, v18.8b, v8.8b, #6\n\t"
                "ext v9.8b, v19.8b, v9.8b, #6\n\t"
                "smull v20.4s, v16.4h, v14.h[0]\n\t"
                "smull v21.4s, v17.4h, v14.h[2]\n\t"
                "smull v22.4s, v18.4h, v15.h[0]\n\t"
                "smull v23.4s, v19.4h, v15.h[2]\n\t"
                "smlal v20.4s, v6.4h, v14.h[1]\n\t"
                "smlal v21.4s, v7.4h, v14.h[3]\n\t"
                "smlal v22.4s, v8.4h, v15.h[1]\n\t"
                "smlal v23.4s, v9.4h, v15.h[3]\n\t"
                // results calculation
                "add v10.4s, v10.4s, v11.4s\n\t"
                "add v12.4s, v12.4s, v13.4s\n\t"
                "add v20.4s, v20.4s, v21.4s\n\t"
                "add v22.4s, v22.4s, v23.4s\n\t"
                "add v10.4s, v10.4s, %9.4s\n\t"
                "add v12.4s, v12.4s, %9.4s\n\t"
                "add v20.4s, v20.4s, %9.4s\n\t"
                "add v22.4s, v22.4s, %9.4s\n\t"
                "shrn v10.4h, v10.4s, #15\n\t"
                "shrn v11.4h, v12.4s, #15\n\t"
                "shrn v12.4h, v20.4s, #15\n\t"
                "shrn v13.4h, v22.4s, #15\n\t"
                "ins v10.d[1], v11.d[0]\n\t"
                "ins v12.d[1], v13.d[0]\n\t"
                "sqxtun v10.8b, v10.8h\n\t"
                "sqxtun v12.8b, v12.8h\n\t"
                "subs x25, x25, #1\n\t"
                "tbl v10.8b, {v10.16b}, %10.8b\n\t"
                "tbl v12.8b, {v12.16b}, %10.8b\n\t"
                "st1 {v10.s}[0], [%0], #4\n\t"
                "st1 {v10.h}[2], [%0], #2\n\t"
                "st1 {v12.s}[0], [%0], #4\n\t"
                "st1 {v12.h}[2], [%0], #2\n\t"
                "bge 0b\n\t"
                "1:\n\t"
                : "=r"(dst_loc)
                : "r"(dst_loc), "r"(simd_loop), "r"(x), "r"(tab_loc), "r"(buf_loc), "r"(BilinearTab_ptr), "r"(src),
                  "r"(src2), "w"(DELTA_vec), "w"(tb), "r"(cmp_flag)
                : "cc", "memory", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "v0", "v1", "v2", "v3",
                  "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                  "v19", "v20", "v21", "v22", "v23");
            x = x + (simd_loop << 2);
        }
#else
        short* BilinearTab_ptr = BilinearTab_i[0][0];
        int* buf_loc_base      = &buf_loc[x];
        short* tab_loc_base    = &tab_loc[x];
        for (; x <= end_x - 4 + 1; x += 4) {
            asm volatile(
                "#load from tab_loc and buf_loc\n\t"
                // "add r7, %1, %7, lsl #1\n\t"
                // "add r8, %2, %7, lsl #2\n\t"
                "vld1.16 {d0}, [%1]!\n\t"
                "vld1.32 {q1}, [%2]!\n\t"

                "vmov.s16 r1, d0[0]\n\t"
                "vmov.s16 r4, d0[1]\n\t"
                "vmov.32 r2, d2[0]\n\t"
                // "vmov.32 r11, d2[1]\n\t"
                "add r1, %6, r1, lsl #3\n\t"
                "add r4, %6, r4, lsl #3\n\t"
                "add r3, %8, r2\n\t"
                // "add r12, %5, r11\n\t"
                "add r2, %7, r2\n\t"
                // "add r11, %4, r11\n\t"
                "#wtab0 and wtab1\n\t"
                "vld1.16 {d4}, [r1]\n\t"
                "vmov.32 r1, d2[1]\n\t"  // new
                "vld1.16 {d1}, [r4]\n\t"
                "add r4, %8, r1\n\t"  // new
                "add r1, %7, r1\n\t"  // new
                "#point_vec00 and point_vec01\n\t"
                "vld1.8 {d6}, [r2]\n\t"
                "vld1.8 {d8}, [r3]\n\t"
                "#point_vec10 and point_vec10\n\t"
                "vld1.8 {d12}, [r1]\n\t"  // new
                "vld1.8 {d14}, [r4]\n\t"  // new

                "vmov.s16 r1, d0[2]\n\t"
                "vmov.s16 r4, d0[3]\n\t"
                "vmov.32 r2, d3[0]\n\t"
                // "vmov.32 r11, d3[1]\n\t"

                "#calculate vec00 and vec01\n\t"
                "vmovl.u8 q3, d6\n\t"
                "vmovl.u8 q4, d8\n\t"
                "#calculate vec10 and vec11\n\t"
                "vmovl.u8 q6, d12\n\t"
                "vmovl.u8 q7, d14\n\t"

                "add r1, %6, r1, lsl #3\n\t"
                "add r4, %6, r4, lsl #3\n\t"
                "add r3, %8, r2\n\t"
                // "add r12, %5, r11\n\t"
                "add r2, %7, r2\n\t"
                // "add r11, %4, r11\n\t"

                "vext.8 d7, d6, d7, #6\n\t"
                "vext.8 d9, d8, d9, #6\n\t"
                "vext.8 d13, d12, d13, #6\n\t"
                "vext.8 d15, d14, d15, #6\n\t"

                "vmull.s16 q10, d6, d4[0]\n\t"
                "vmull.s16 q11, d8, d4[2]\n\t"
                "vmlal.s16 q10, d7, d4[1]\n\t"
                "vmlal.s16 q11, d9, d4[3]\n\t"

                "#wtab2\n\t"
                "vld1.16 {d4}, [r1]\n\t"
                "vmov.32 r1, d3[1]\n\t"  // new
                "#point_vec20 and point_vec21\n\t"
                "vld1.8 {d6}, [r2]\n\t"
                "vld1.8 {d8}, [r3]\n\t"

                "vmull.s16 q12, d12, d1[0]\n\t"
                "vmull.s16 q13, d14, d1[2]\n\t"
                "vmlal.s16 q12, d13, d1[1]\n\t"
                "vmlal.s16 q13, d15, d1[3]\n\t"
                "vadd.i32 q10, q11, q10\n\t"
                "vadd.i32 q12, q12, q13\n\t"

                "#wtab3\n\t"
                "vld1.16 {d1}, [r4]\n\t"
                "add r4, %8, r1\n\t"  // new
                "add r1, %7, r1\n\t"  // new
                "#point_vec30 and point_vec31\n\t"
                "vld1.8 {d12}, [r1]\n\t"  // new
                "vld1.8 {d14}, [r4]\n\t"  // new

                "vadd.i32 q10, q10, %q9\n\t"
                "vadd.i32 q12, q12, %q9\n\t"
                "vshrn.i32 d20, q10, #15\n\t"
                "vshrn.i32 d21, q12, #15\n\t"

                "#calculate vec20 and vec21\n\t"
                "vmovl.u8 q3, d6\n\t"
                "vmovl.u8 q4, d8\n\t"

                "#store results\n\t"
                "vmovn.i16 d20, q10\n\t"
                "vtbl.8 d20, {d20}, %10\n\t"

                "#calculate vec30 and vec31\n\t"
                "vmovl.u8 q6, d12\n\t"
                "vmovl.u8 q7, d14\n\t"

                "vext.8 d7, d6, d7, #6\n\t"
                "vext.8 d9, d8, d9, #6\n\t"
                "vext.8 d13, d12, d13, #6\n\t"
                "vext.8 d15, d14, d15, #6\n\t"

                "vst1.32 {d20[0]}, [%0]!\n\t"
                "vst1.16 {d20[2]}, [%0]!\n\t"

                "vmull.s16 q10, d6, d4[0]\n\t"
                "vmull.s16 q11, d8, d4[2]\n\t"
                "vmlal.s16 q10, d7, d4[1]\n\t"
                "vmlal.s16 q11, d9, d4[3]\n\t"
                "vmull.s16 q12, d12, d1[0]\n\t"
                "vmull.s16 q13, d14, d1[2]\n\t"
                "vmlal.s16 q12, d13, d1[1]\n\t"
                "vmlal.s16 q13, d15, d1[3]\n\t"

                "vadd.i32 q10, q11, q10\n\t"
                "vadd.i32 q12, q12, q13\n\t"
                "vadd.i32 q10, q10, %q9\n\t"
                "vadd.i32 q12, q12, %q9\n\t"
                "vshrn.i32 d20, q10, #15\n\t"
                "vshrn.i32 d21, q12, #15\n\t"
                "vmovn.i16 d20, q10\n\t"
                "vtbl.8 d20, {d20}, %10\n\t"

                "vst1.32 {d20[0]}, [%0]!\n\t"
                "vst1.16 {d20[2]}, [%0]!\n\t"
                : "=r"(dst_loc), "=r"(tab_loc_base), "=r"(buf_loc_base)
                : "0"(dst_loc), "1"(tab_loc_base), "2"(buf_loc_base), "r"(BilinearTab_ptr), "r"(src), "r"(src2),
                  "w"(DELTA_vec), "w"(tb)
                : "cc", "memory", "r1", "r2", "r3", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q10", "q11",
                  "q12", "q13");
            // dst_loc += 12;
        }
#endif  // a64
#endif  // end of NEON

        for (; x <= end_x; x++) {
            // int final_loc = (y * dst_w + x)*schannel;
            int final_loc  = final_loc_base + x * 3;
            int loc_buffer = buf_loc[x];
            // int loc_buffer = xy_loc_buf[2 * x] * 3 + xy_loc_buf[2 * x + 1] * stmp;
            short* wtab = BilinearTab_i[tab_loc[x]][0];
            // short *wtab = BilinearTab_i[xy_float_buf[x]][0];

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
#endif
}

}  // namespace TNN_NS
