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

#include "stdlib.h"
#include <algorithm>
#include <type_traits>

#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

#include "tnn/core/macro.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/mat_converter_utils.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

static inline void *armMalloc(size_t size) {
#if _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void *ptr = 0;
    if (posix_memalign(&ptr, 32, size))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(32, size);
#else
    return malloc(size);
#endif
}


#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), (int)0), (int)UCHAR_MAX)
#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), (int)SHRT_MIN), (int)SHRT_MAX)
#define SATURATE_CAST_INT(X) (int)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), (int)INT_MIN), (int)INT_MAX)

void MatMemcpy2D(void* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    auto src_ptr = reinterpret_cast<uint8_t*>(src);
    auto dst_ptr = reinterpret_cast<uint8_t*>(dst);

    for (int h = 0; h < height; h++) {
        memcpy(dst_ptr, src_ptr, width);
        src_ptr += src_stride;
        dst_ptr += dst_stride;
    }

}

void MatMemcpy2DWithPadding(void* src, void* dst, int width, int height, int src_stride, int dst_stride,
                            int top, int bottom, int left, int right, uint8_t pad_val) {
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

static void ResizeGetAdjacentRows(int sy, int prev_sy, short** rows0, short** rows1, int* xofs, 
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

static void ResizeCalculateOneRow(short* rows0p, short* rows1p, const int b0, const int b1, const int w, const int c,
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

struct ResizeBilinearKernelParm {
    ResizeBilinearKernelParm(int* _xofs, int* _yofs, short* _ialpha, short* _ibeta,
                             const uint8_t* _src, uint8_t* _dst, int _src_plane, int _src_stride, int _schannel) {
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

void ResizeBilinearOneRow(ResizeBilinearKernelParm& param, int thread_id, short** rows0_t, short** rows1_t,
                          int* prev_sy, int b, int w, int h, int stride, int dy) {
    int sy = param.yofs[dy];
    ResizeGetAdjacentRows(sy, prev_sy[thread_id], &rows0_t[thread_id], &rows1_t[thread_id], param.xofs,
                          param.src + b * param.src_plane, param.src_stride, param.schannel, w, param.ialpha);
    prev_sy[thread_id] = sy;

    // vresize
    short b0 = param.ibeta[dy * 2];
    short b1 = param.ibeta[dy * 2 + 1];

    uint8_t* Dp   = param.dst + stride * (b * h + dy);

    ResizeCalculateOneRow(rows0_t[thread_id], rows1_t[thread_id], b0, b1, w, param.schannel, Dp);
}

#define ResizeBilinearPreparation(channel)                               \
    int schannel  = channel;                                             \
    int* buf      = nullptr;                                             \
    GetResizeBuf(src_w, src_h, w, h, schannel, &buf);                    \
    int* xofs     = buf;                                                 \
    int* yofs     = buf + w;                                             \
    short* ialpha = (short*)(buf + w + h);                               \
    short* ibeta  = (short*)(buf + w + h + w);                           \
    int src_plane = src_h * src_stride;

void ResizeBilinearC1Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride,
                          uint8_t* dst, int w, int h, int stride) {
    ResizeBilinearPreparation(1);

    ResizeBilinearKernelParm param(xofs, yofs, ialpha, ibeta, src, dst, src_plane, src_stride, schannel);

    // loop body
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    short* rows0        = new short[w * max_num_threads];
    short* rows1        = new short[w * max_num_threads];
    short* rows0_t[max_num_threads];
    short* rows1_t[max_num_threads];
    int    prev_sy[max_num_threads];

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < max_num_threads; ++t) {
            prev_sy[t] = -2;
            rows0_t[t] = rows0 + t * w;
            rows1_t[t] = rows1 + t * w;
        }

        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            int thread_id  = OMP_TID_;
            ResizeBilinearOneRow(param, thread_id, rows0_t, rows1_t, prev_sy, b, w, h, stride, dy);
        }
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
}

void ResizeBilinearC2Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride,
                          uint8_t* dst, int w, int h, int stride) {
    ResizeBilinearPreparation(2);

    ResizeBilinearKernelParm param(xofs, yofs, ialpha, ibeta, src, dst, src_plane, src_stride, schannel);

    // loop body
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    short* rows0        = new short[(w * 2 + 2) * max_num_threads];
    short* rows1        = new short[(w * 2 + 2) * max_num_threads];
    short* rows0_t[max_num_threads];
    short* rows1_t[max_num_threads];
    int    prev_sy[max_num_threads];

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < max_num_threads; ++t) {
            prev_sy[t] = -2;
            rows0_t[t] = rows0 + t * (w * 2 + 2);
            rows1_t[t] = rows1 + t * (w * 2 + 2);
        }

        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            int thread_id  = OMP_TID_;
            ResizeBilinearOneRow(param, thread_id, rows0_t, rows1_t, prev_sy, b, w, h, stride, dy);
        }
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
}

void ResizeBilinearC3Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride,
                          uint8_t* dst, int w, int h, int stride) {
    ResizeBilinearPreparation(3);

    ResizeBilinearKernelParm param(xofs, yofs, ialpha, ibeta, src, dst, src_plane, src_stride, schannel);

    // loop body
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    short* rows0        = new short[(w * 3 + 1) * max_num_threads];
    short* rows1        = new short[(w * 3 + 1) * max_num_threads];
    short* rows0_t[max_num_threads];
    short* rows1_t[max_num_threads];
    int    prev_sy[max_num_threads];

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < max_num_threads; ++t) {
            prev_sy[t] = -2;
            rows0_t[t] = rows0 + t * (w * 3 + 1);
            rows1_t[t] = rows1 + t * (w * 3 + 1);
        }

        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            int thread_id  = OMP_TID_;
            ResizeBilinearOneRow(param, thread_id, rows0_t, rows1_t, prev_sy, b, w, h, stride, dy);
        }
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
}

void ResizeBilinearC4Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride,
                          uint8_t* dst, int w, int h, int stride) {
    ResizeBilinearPreparation(4);

    ResizeBilinearKernelParm param(xofs, yofs, ialpha, ibeta, src, dst, src_plane, src_stride, schannel);

    // loop body
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    short* rows0        = new short[(w * 4) * max_num_threads];
    short* rows1        = new short[(w * 4) * max_num_threads];
    short* rows0_t[max_num_threads];
    short* rows1_t[max_num_threads];
    int    prev_sy[max_num_threads];

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < max_num_threads; ++t) {
            prev_sy[t] = -2;
            rows0_t[t] = rows0 + t * (w * 4);
            rows1_t[t] = rows1 + t * (w * 4);
        }

        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            int thread_id  = OMP_TID_;
            ResizeBilinearOneRow(param, thread_id, rows0_t, rows1_t, prev_sy, b, w, h, stride, dy);
        }
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
}

void ResizeBilinearYUV420sp(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h) {
    // assert src_w % 2 == 0
    // assert src_h % 2 == 0
    // assert w % 2 == 0
    // assert h % 2 == 0

    int src_plane = src_w * src_h * 3 / 2;
    int dst_plane = w * h * 3 / 2;

    for (int b = 0; b < batch; ++b) {
        const uint8_t* srcY  = src + b * src_plane;
        uint8_t* dstY        = dst + b * dst_plane;
        ResizeBilinearC1(srcY, 1, src_w, src_h, dstY, w, h);

        const uint8_t* srcUV = srcY + src_w * src_h;
        uint8_t* dstUV       = dstY + w * h;
        ResizeBilinearC2(srcUV, 1, src_w / 2, src_h / 2, dstUV, w / 2, h / 2);
    }
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

#define ResizeNearestPreparation(channel)                               \
    int schannel  = channel;                                            \
    int* buf      = nullptr;                                            \
    GetResizeBufNearset(src_w, src_h, w, h, schannel, &buf);            \
    int* xofs     = buf;                                                \
    int* yofs     = buf + w;                                            \
    uint8_t* ialpha = (uint8_t*)(buf + w + h);                          \
    uint8_t* ibeta  = (uint8_t*)(buf + w + h + w);

#define ResizeNearestLoopPreparation()                                  \
    int sy = (ibeta[dy] == 0) ? yofs[dy] + 1 : yofs[dy];                \
    const uint8_t* Sp = src + src_stride * (b * src_h + sy);            \
    uint8_t* Dp       = dst + stride * (b * h + dy);                    \
    int dx = 0;                                                         \

#ifdef TNN_USE_NEON

#define MAKE_LOAD(n)                                                    \
    _sx = vld1q_s32(xofs_p);                                            \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 0), _S0, 0);        \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 1), _S0, 1);        \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 2), _S0, 2);        \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 3), _S0, 3);        \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 0) + n, _S1, 0);    \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 1) + n, _S1, 1);    \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 2) + n, _S1, 2);    \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 3) + n, _S1, 3);    \
    _sx = vld1q_s32(xofs_p + 4);                                        \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 0), _S0, 4);        \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 1), _S0, 5);        \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 2), _S0, 6);        \
    _S0 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 3), _S0, 7);        \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 0) + n, _S1, 4);    \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 1) + n, _S1, 5);    \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 2) + n, _S1, 6);    \
    _S1 = vld##n##_lane_u8(Sp + vgetq_lane_s32(_sx, 3) + n, _S1, 7);    \
    uint8x8_t _mask = vld1_u8(ialpha_p);                                \

#define LOAD_C1() MAKE_LOAD(1)
#define LOAD_C2() MAKE_LOAD(2)
#define LOAD_C3() MAKE_LOAD(3)
#define LOAD_C4() MAKE_LOAD(4)

#endif  // TNN_USE_NEON

void ResizeNearestC1Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride,
                         uint8_t* dst, int w, int h, int stride) {
    ResizeNearestPreparation(1);

    // loop body
    for (int b = 0; b < batch; ++b) {
        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            ResizeNearestLoopPreparation();
#ifdef TNN_USE_NEON
            int32x4_t _sx = int32x4_t();
            uint8x8_t _S0 = uint8x8_t();
            uint8x8_t _S1 = uint8x8_t();
            int* xofs_p       = xofs;
            uint8_t* ialpha_p = ialpha;
            uint8_t* Dp_p     = Dp;
            int simd_loop = 0;
            for (int i = 0; i < w - 7; i += 8) {
                LOAD_C1();

                vst1_u8(Dp_p, vbsl_u8(_mask, _S0, _S1));

                xofs_p   += 8;
                ialpha_p += 8;
                Dp_p     += 8;
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

void ResizeNearestC2Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride,
                         uint8_t* dst, int w, int h, int stride) {
    ResizeNearestPreparation(2);

    // loop body
    for (int b = 0; b < batch; ++b) {
        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            ResizeNearestLoopPreparation();
#ifdef TNN_USE_NEON
            int32x4_t _sx   = int32x4_t();
            uint8x8x2_t _S0 = uint8x8x2_t();
            uint8x8x2_t _S1 = uint8x8x2_t();
            uint8x8x2_t _S2 = uint8x8x2_t();
            int* xofs_p       = xofs;
            uint8_t* ialpha_p = ialpha;
            uint8_t* Dp_p     = Dp;
            int simd_loop  = 0;
            for (int i = 0; i < w - 7; i += 8) {
                LOAD_C2();

                _S2.val[0] = vbsl_u8(_mask, _S0.val[0], _S1.val[0]);
                _S2.val[1] = vbsl_u8(_mask, _S0.val[1], _S1.val[1]);
                vst2_u8(Dp_p, _S2);

                xofs_p   += 8;
                ialpha_p += 8;
                Dp_p     += 8 * 2;
                ++simd_loop;
            }
            dx += simd_loop * 8;
#endif
            for (; dx < w; dx++) {
                int sx = xofs[dx];
                Dp[dx * 2]     = (ialpha[dx] == 0) ? Sp[sx + 2] : Sp[sx];
                Dp[dx * 2 + 1] = (ialpha[dx] == 0) ? Sp[sx + 3] : Sp[sx + 1];
            }
        }
    }

    delete[] buf;
}

void ResizeNearestC3Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride,
                         uint8_t* dst, int w, int h, int stride) {
    ResizeNearestPreparation(3);

    // loop body
    for (int b = 0; b < batch; ++b) {
        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            ResizeNearestLoopPreparation();
#ifdef TNN_USE_NEON
            int32x4_t _sx   = int32x4_t();
            uint8x8x3_t _S0 = uint8x8x3_t();
            uint8x8x3_t _S1 = uint8x8x3_t();
            uint8x8x3_t _S2 = uint8x8x3_t();
            int* xofs_p       = xofs;
            uint8_t* ialpha_p = ialpha;
            uint8_t* Dp_p     = Dp;
            int simd_loop = 0;
            for (int i = 0; i < w - 7; i += 8) {
                LOAD_C3();

                _S2.val[0] = vbsl_u8(_mask, _S0.val[0], _S1.val[0]);
                _S2.val[1] = vbsl_u8(_mask, _S0.val[1], _S1.val[1]);
                _S2.val[2] = vbsl_u8(_mask, _S0.val[2], _S1.val[2]);
                vst3_u8(Dp_p, _S2);

                xofs_p   += 8;
                ialpha_p += 8;
                Dp_p     += 8 * 3;
                ++simd_loop;
            }
            dx += simd_loop * 8;
#endif
            for (; dx < w; dx++) {
                int sx = xofs[dx];
                Dp[dx * 3]     = (ialpha[dx] == 0) ? Sp[sx + 3] : Sp[sx];
                Dp[dx * 3 + 1] = (ialpha[dx] == 0) ? Sp[sx + 4] : Sp[sx + 1];
                Dp[dx * 3 + 2] = (ialpha[dx] == 0) ? Sp[sx + 5] : Sp[sx + 2];
            }
        }
    }

    delete[] buf;
}

void ResizeNearestC4Impl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride,
                         uint8_t* dst, int w, int h, int stride) {
    ResizeNearestPreparation(4);

    // loop body
    for (int b = 0; b < batch; ++b) {
        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            ResizeNearestLoopPreparation();
#ifdef TNN_USE_NEON
            int32x4_t _sx   = int32x4_t();
            uint8x8x4_t _S0 = uint8x8x4_t();
            uint8x8x4_t _S1 = uint8x8x4_t();
            uint8x8x4_t _S2 = uint8x8x4_t();
            int* xofs_p       = xofs;
            uint8_t* ialpha_p = ialpha;
            uint8_t* Dp_p     = Dp;
            int simd_loop = 0;
            for (int i = 0; i < w - 7; i += 8) {
                LOAD_C4();

                _S2.val[0] = vbsl_u8(_mask, _S0.val[0], _S1.val[0]);
                _S2.val[1] = vbsl_u8(_mask, _S0.val[1], _S1.val[1]);
                _S2.val[2] = vbsl_u8(_mask, _S0.val[2], _S1.val[2]);
                _S2.val[3] = vbsl_u8(_mask, _S0.val[3], _S1.val[3]);
                vst4_u8(Dp_p, _S2);

                xofs_p   += 8;
                ialpha_p += 8;
                Dp_p     += 8 * 4;
                ++simd_loop;
            }
            dx += simd_loop * 8;
#endif
            for (; dx < w; dx++) {
                int sx = xofs[dx];
                Dp[dx * 4]     = (ialpha[dx] == 0) ? Sp[sx + 4] : Sp[sx];
                Dp[dx * 4 + 1] = (ialpha[dx] == 0) ? Sp[sx + 5] : Sp[sx + 1];
                Dp[dx * 4 + 2] = (ialpha[dx] == 0) ? Sp[sx + 6] : Sp[sx + 2];
                Dp[dx * 4 + 3] = (ialpha[dx] == 0) ? Sp[sx + 7] : Sp[sx + 3];
            }
        }
    }

    delete[] buf;
}

#ifdef TNN_USE_NEON

#undef MAKE_LOAD
#undef LOAD_C1
#undef LOAD_C2
#undef LOAD_C3
#undef LOAD_C4

#endif  // TNN_USE_NEON

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
        const uint8_t* srcY  = src + b * src_plane;
        uint8_t* dstY        = dst + b * dst_plane;
        ResizeNearestC1(srcY, 1, src_w, src_h, dstY, w, h);

        const uint8_t* srcUV = srcY + src_w * src_h;
        uint8_t* dstUV       = dstY + w * h;
        ResizeNearestC2(srcUV, 1, src_w / 2, src_h / 2, dstUV, w / 2, h / 2);
    }
}

#define INTER_REMAP_COEF_BITS  15
#define INTER_REMAP_COEF_SCALE (1<<INTER_REMAP_COEF_BITS)
#define INTER_BITS      5
#define INTER_TAB_SIZE  (1<<INTER_BITS)
#define KSIZE 2
static short BilinearTab_i[INTER_TAB_SIZE*INTER_TAB_SIZE][KSIZE][KSIZE];

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
    int ksize = KSIZE;

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

    *buffer = reinterpret_cast<int*>(armMalloc((dst_w + dst_h) * 2 * sizeof(int)));

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
    return new_x_loc >= -1 && new_x_loc <= (src_w - 1) &&
           new_y_loc >= -1 && new_y_loc <= (src_h - 1);
}

static inline bool CheckDataIsInBoundary(const int new_x_loc, const int new_y_loc, const int src_w, const int src_h) {
    return new_x_loc >= 0 && new_x_loc < (src_w - 1) &&
           new_y_loc >= 0 && new_y_loc < (src_h - 1);
}

static void WarpAffinePrepareOneRow(int* buf_loc, short* tab_loc, int* adelta, int* bdelta, int channel,
                                    const uint8_t* src, int src_w, int src_h, uint8_t* dst, int dst_w,
                                    int y, int src_offset, int& x_count, int& end_x, float border_val = 0) {
    const unsigned char* src2 = src + src_w * channel;

    short xy_loc_buf[dst_w * 2];
    short tb_loc_buf[dst_w];
    int   sc_loc_buf[dst_w];
    int*   adelta_p     = adelta;
    short* xy_loc_buf_p = xy_loc_buf;
    short* tb_loc_buf_p = tb_loc_buf;
    int*   sc_loc_buf_p = sc_loc_buf;
    int x = 0;
#ifdef TNN_USE_NEON
    int32x2_t _bdelta0 = vld1_s32(bdelta + 2 * y);
    int32x4_t _bdelta  = vcombine_s32(_bdelta0, _bdelta0);
    int32x4_t _offset  = vdupq_n_s32(16);
    int16x8_t _mask    = vdupq_n_s16(31);
    int16x8_t _coeff   = {1,32,1,32,1,32,1,32};
    int32x4_t _channel = vdupq_n_s32(channel);
    int32x4_t _soffset = vdupq_n_s32(src_offset);
    int16x4_t _src_w   = {1, (short)src_w,1,(short)src_w};
    for (; x < dst_w>>2<<2; x += 4) {
        int32x4_t _xyxy0   = vaddq_s32(vld1q_s32(adelta_p), _offset);
        int32x4_t _xyxy1   = vaddq_s32(vld1q_s32(adelta_p + 4), _offset);
        _xyxy0             = vaddq_s32(_xyxy0, _bdelta);
        _xyxy1             = vaddq_s32(_xyxy1, _bdelta);
        int16x4_t _xyxy0s  = vshrn_n_s32(_xyxy0, 10);
        //int16x8_t _xyxy01s = vshrn_high_n_s32(_xyxy0s, _xyxy1, 10);
        int16x4_t _xyxy1s  = vshrn_n_s32(_xyxy1, 10);
        int16x8_t _xyxy01s = vcombine_s16(_xyxy0s, _xyxy1s);
        vst1q_s16(xy_loc_buf_p, _xyxy01s);

        int32x4_t _src_0   = vmull_s16(_xyxy0s, _src_w);
        int32x4_t _src_1   = vmull_s16(vget_high_s16(_xyxy01s), _src_w);
        vst1q_s32(sc_loc_buf_p, vaddq_s32(vmulq_s32(_channel, VPADDQ_S32(_src_0, _src_1)), _soffset));

        _xyxy0s            = vshrn_n_s32(_xyxy0, 5);
        //_xyxy01s           = vshrn_high_n_s32(_xyxy0s, _xyxy1, 5);
        _xyxy1s            = vshrn_n_s32(_xyxy1, 5);
        _xyxy01s           = vcombine_s16(_xyxy0s, _xyxy1s);
        int16x8_t _tab_xys = vmulq_s16(vandq_s16(_xyxy01s, _mask), _coeff);
        vst1_s16(tb_loc_buf_p, vpadd_s16(vget_low_s16(_tab_xys), vget_high_s16(_tab_xys)));

        adelta_p     += 8;
        xy_loc_buf_p += 8;
        tb_loc_buf_p += 4;
        sc_loc_buf_p += 4;
    }
    if (dst_w % 4) {
        x -= 4;
    }
#endif
    for (; x < dst_w; ++x) {
        int new_x     = adelta[2 * x] + bdelta[2 * y] + 16;
        int new_y     = adelta[2 * x + 1] + bdelta[2 * y + 1] + 16;
        int new_x_loc = new_x >> 10;
        int new_y_loc = new_y >> 10;
        xy_loc_buf[2 * x]     = new_x_loc;
        xy_loc_buf[2 * x + 1] = new_y_loc;
        tb_loc_buf[x] = ((new_x >> 5) & 31) + ((new_y >> 5) & 31) * 32;
        sc_loc_buf[x] = (new_x_loc + new_y_loc * src_w) * channel + src_offset;
    }

    for (x = 0; x < dst_w; ++x) {
        short new_x_loc    = xy_loc_buf[2 * x];
        short new_y_loc    = xy_loc_buf[2 * x + 1];
        short new_xy_float = tb_loc_buf[x];
        int   src_loc      = sc_loc_buf[x];

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
}

static void WarpAffineCalculateOneRow(int begin_x, int end_x, int channel, int dst_loc_base, const int* buf_loc,
                                      const short* tab_loc, const uint8_t* src1, const uint8_t* src2, uint8_t* dst) {
    const int* buf_loc_p   = buf_loc + begin_x;
    const short* tab_loc_p = tab_loc + begin_x;
    const short* tab_p     = BilinearTab_i[0][0];
    int x                  = begin_x;

#ifdef TNN_USE_NEON

#define MAKE_CAL(n)                                                                 \
    _val0    = vmull_s16(_tab0, vget_low_s16(_src16_0##n));                         \
    _val1    = vmull_s16(_tab1, vget_high_s16(_src16_0##n));                        \
    _val2    = vmull_s16(_tab2, vget_low_s16(_src16_1##n));                         \
    _val3    = vmull_s16(_tab3, vget_high_s16(_src16_1##n));                        \
    _res0123 = VPADDQ_S32(VPADDQ_S32(_val0, _val1), VPADDQ_S32(_val2, _val3));      \
    _res0123 = vaddq_s32(_res0123, _offset);                                        \
    _res16   = vshrn_n_s32(_res0123, 15);                                           \
    _resu8.val[n] = vqmovun_s16(vcombine_s16(_res16, _res16));                      \

#define CAL_C0() MAKE_CAL(0)
#define CAL_C1() MAKE_CAL(1)
#define CAL_C2() MAKE_CAL(2)
#define CAL_C3() MAKE_CAL(3)

#endif

    if (channel == 1) {
#ifdef TNN_USE_NEON
        uint8_t* dst_p         = dst +  dst_loc_base + begin_x * 1;
        int32x4_t _offset      = vdupq_n_s32(1 << 14);
        int simd_loop          = 0;
        for (; x <= end_x - 3; x += 4) {
            int32x4_t _src_loc = vld1q_s32(buf_loc_p);
            uint8x8_t _src01;
            _src01 = vld1_lane_u8(src1 + vgetq_lane_s32(_src_loc, 0), _src01, 0);
            _src01 = vld1_lane_u8(src1 + vgetq_lane_s32(_src_loc, 0) + 1, _src01, 1);
            _src01 = vld1_lane_u8(src2 + vgetq_lane_s32(_src_loc, 0), _src01, 2);
            _src01 = vld1_lane_u8(src2 + vgetq_lane_s32(_src_loc, 0) + 1, _src01, 3);
            _src01 = vld1_lane_u8(src1 + vgetq_lane_s32(_src_loc, 1), _src01, 4);
            _src01 = vld1_lane_u8(src1 + vgetq_lane_s32(_src_loc, 1) + 1, _src01, 5);
            _src01 = vld1_lane_u8(src2 + vgetq_lane_s32(_src_loc, 1), _src01, 6);
            _src01 = vld1_lane_u8(src2 + vgetq_lane_s32(_src_loc, 1) + 1, _src01, 7);
            int16x8_t _src16_0 = vreinterpretq_s16_u16(vmovl_u8(_src01));
            uint8x8_t _src23;
            _src23 = vld1_lane_u8(src1 + vgetq_lane_s32(_src_loc, 2), _src23, 0);
            _src23 = vld1_lane_u8(src1 + vgetq_lane_s32(_src_loc, 2) + 1, _src23, 1);
            _src23 = vld1_lane_u8(src2 + vgetq_lane_s32(_src_loc, 2), _src23, 2);
            _src23 = vld1_lane_u8(src2 + vgetq_lane_s32(_src_loc, 2) + 1, _src23, 3);
            _src23 = vld1_lane_u8(src1 + vgetq_lane_s32(_src_loc, 3), _src23, 4);
            _src23 = vld1_lane_u8(src1 + vgetq_lane_s32(_src_loc, 3) + 1, _src23, 5);
            _src23 = vld1_lane_u8(src2 + vgetq_lane_s32(_src_loc, 3), _src23, 6);
            _src23 = vld1_lane_u8(src2 + vgetq_lane_s32(_src_loc, 3) + 1, _src23, 7);
            int16x8_t _src16_1 = vreinterpretq_s16_u16(vmovl_u8(_src23));

            int16x4_t _tab_loc = vld1_s16(tab_loc_p);
            int16x4_t _tab0    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 0) * 4);
            int16x4_t _tab1    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 1) * 4);
            int16x4_t _tab2    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 2) * 4);
            int16x4_t _tab3    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 3) * 4);

            int32x4_t _val0    = vmull_s16(_tab0, vget_low_s16(_src16_0));
            int32x4_t _val1    = vmull_s16(_tab1, vget_high_s16(_src16_0));
            int32x4_t _val2    = vmull_s16(_tab2, vget_low_s16(_src16_1));
            int32x4_t _val3    = vmull_s16(_tab3, vget_high_s16(_src16_1));

            int32x4_t _res0123 = VPADDQ_S32(VPADDQ_S32(_val0, _val1), VPADDQ_S32(_val2, _val3));
            _res0123           = vaddq_s32(_res0123, _offset);
            int16x4_t _res16   = vshrn_n_s32(_res0123, 15);
            uint8x8_t _resu8   = vqmovun_s16(vcombine_s16(_res16, _res16));

            vst1_lane_u8(dst_p, _resu8, 0);
            vst1_lane_u8(dst_p + 1, _resu8, 1);
            vst1_lane_u8(dst_p + 2, _resu8, 2);
            vst1_lane_u8(dst_p + 3, _resu8, 3);

            buf_loc_p += 4;
            tab_loc_p += 4;
            dst_p     += 4;
            ++simd_loop;
        }
        x = begin_x + (simd_loop << 2);
#endif
        for (; x <= end_x; x++) {
            int dst_loc = dst_loc_base + x * 1;
            int src_loc = buf_loc[x];
            short* wtab = BilinearTab_i[tab_loc[x]][0];

            int point0 = src1[src_loc];
            int point1 = src1[src_loc + 1];
            int point2 = src2[src_loc];
            int point3 = src2[src_loc + 1];

            int val_xy0  = wtab[0] * point0 + wtab[1] * point1 + wtab[2] * point2 + wtab[3] * point3;
            dst[dst_loc] = SATURATE_CAST_UCHAR((val_xy0 + (1 << 14)) >> 15);
        }
    } else if (channel == 2) {
#ifdef TNN_USE_NEON
        uint8_t* dst_p         = dst +  dst_loc_base + begin_x * 2;
        int32x4_t _offset      = vdupq_n_s32(1 << 14);
        int simd_loop          = 0;
        for (; x <= end_x - 3; x += 4) {
            int32x4_t _src_loc = vld1q_s32(buf_loc_p);
            uint8x8x2_t _src01;
            _src01 = vld2_lane_u8(src1 + vgetq_lane_s32(_src_loc, 0), _src01, 0);
            _src01 = vld2_lane_u8(src1 + vgetq_lane_s32(_src_loc, 0) + 2, _src01, 1);
            _src01 = vld2_lane_u8(src2 + vgetq_lane_s32(_src_loc, 0), _src01, 2);
            _src01 = vld2_lane_u8(src2 + vgetq_lane_s32(_src_loc, 0) + 2, _src01, 3);
            _src01 = vld2_lane_u8(src1 + vgetq_lane_s32(_src_loc, 1), _src01, 4);
            _src01 = vld2_lane_u8(src1 + vgetq_lane_s32(_src_loc, 1) + 2, _src01, 5);
            _src01 = vld2_lane_u8(src2 + vgetq_lane_s32(_src_loc, 1), _src01, 6);
            _src01 = vld2_lane_u8(src2 + vgetq_lane_s32(_src_loc, 1) + 2, _src01, 7);
            int16x8_t _src16_00 = vreinterpretq_s16_u16(vmovl_u8(_src01.val[0]));
            int16x8_t _src16_01 = vreinterpretq_s16_u16(vmovl_u8(_src01.val[1]));
            uint8x8x2_t _src23;
            _src23 = vld2_lane_u8(src1 + vgetq_lane_s32(_src_loc, 2), _src23, 0);
            _src23 = vld2_lane_u8(src1 + vgetq_lane_s32(_src_loc, 2) + 2, _src23, 1);
            _src23 = vld2_lane_u8(src2 + vgetq_lane_s32(_src_loc, 2), _src23, 2);
            _src23 = vld2_lane_u8(src2 + vgetq_lane_s32(_src_loc, 2) + 2, _src23, 3);
            _src23 = vld2_lane_u8(src1 + vgetq_lane_s32(_src_loc, 3), _src23, 4);
            _src23 = vld2_lane_u8(src1 + vgetq_lane_s32(_src_loc, 3) + 2, _src23, 5);
            _src23 = vld2_lane_u8(src2 + vgetq_lane_s32(_src_loc, 3), _src23, 6);
            _src23 = vld2_lane_u8(src2 + vgetq_lane_s32(_src_loc, 3) + 2, _src23, 7);
            int16x8_t _src16_10 = vreinterpretq_s16_u16(vmovl_u8(_src23.val[0]));
            int16x8_t _src16_11 = vreinterpretq_s16_u16(vmovl_u8(_src23.val[1]));

            int16x4_t _tab_loc = vld1_s16(tab_loc_p);
            int16x4_t _tab0    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 0) * 4);
            int16x4_t _tab1    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 1) * 4);
            int16x4_t _tab2    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 2) * 4);
            int16x4_t _tab3    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 3) * 4);

            int32x4_t _val0, _val1, _val2, _val3, _res0123;
            int16x4_t _res16;
            uint8x8x2_t _resu8;

            CAL_C0();
            CAL_C1();

            vst2_lane_u8(dst_p, _resu8, 0);
            vst2_lane_u8(dst_p + 2, _resu8, 1);
            vst2_lane_u8(dst_p + 4, _resu8, 2);
            vst2_lane_u8(dst_p + 6, _resu8, 3);

            buf_loc_p += 4;
            tab_loc_p += 4;
            dst_p     += 8;
            ++simd_loop;
        }
        x = begin_x + (simd_loop << 2);
#endif
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
#ifdef TNN_USE_NEON
        uint8_t* dst_p         = dst +  dst_loc_base + begin_x * 3;
        int32x4_t _offset      = vdupq_n_s32(1 << 14);
        int simd_loop          = 0;
        for (; x <= end_x - 3; x += 4) {
            int32x4_t _src_loc = vld1q_s32(buf_loc_p);
            uint8x8x3_t _src01;
            _src01 = vld3_lane_u8(src1 + vgetq_lane_s32(_src_loc, 0), _src01, 0);
            _src01 = vld3_lane_u8(src1 + vgetq_lane_s32(_src_loc, 0) + 3, _src01, 1);
            _src01 = vld3_lane_u8(src2 + vgetq_lane_s32(_src_loc, 0), _src01, 2);
            _src01 = vld3_lane_u8(src2 + vgetq_lane_s32(_src_loc, 0) + 3, _src01, 3);
            _src01 = vld3_lane_u8(src1 + vgetq_lane_s32(_src_loc, 1), _src01, 4);
            _src01 = vld3_lane_u8(src1 + vgetq_lane_s32(_src_loc, 1) + 3, _src01, 5);
            _src01 = vld3_lane_u8(src2 + vgetq_lane_s32(_src_loc, 1), _src01, 6);
            _src01 = vld3_lane_u8(src2 + vgetq_lane_s32(_src_loc, 1) + 3, _src01, 7);
            int16x8_t _src16_00 = vreinterpretq_s16_u16(vmovl_u8(_src01.val[0]));
            int16x8_t _src16_01 = vreinterpretq_s16_u16(vmovl_u8(_src01.val[1]));
            int16x8_t _src16_02 = vreinterpretq_s16_u16(vmovl_u8(_src01.val[2]));
            uint8x8x3_t _src23;
            _src23 = vld3_lane_u8(src1 + vgetq_lane_s32(_src_loc, 2), _src23, 0);
            _src23 = vld3_lane_u8(src1 + vgetq_lane_s32(_src_loc, 2) + 3, _src23, 1);
            _src23 = vld3_lane_u8(src2 + vgetq_lane_s32(_src_loc, 2), _src23, 2);
            _src23 = vld3_lane_u8(src2 + vgetq_lane_s32(_src_loc, 2) + 3, _src23, 3);
            _src23 = vld3_lane_u8(src1 + vgetq_lane_s32(_src_loc, 3), _src23, 4);
            _src23 = vld3_lane_u8(src1 + vgetq_lane_s32(_src_loc, 3) + 3, _src23, 5);
            _src23 = vld3_lane_u8(src2 + vgetq_lane_s32(_src_loc, 3), _src23, 6);
            _src23 = vld3_lane_u8(src2 + vgetq_lane_s32(_src_loc, 3) + 3, _src23, 7);
            int16x8_t _src16_10 = vreinterpretq_s16_u16(vmovl_u8(_src23.val[0]));
            int16x8_t _src16_11 = vreinterpretq_s16_u16(vmovl_u8(_src23.val[1]));
            int16x8_t _src16_12 = vreinterpretq_s16_u16(vmovl_u8(_src23.val[2]));

            int16x4_t _tab_loc = vld1_s16(tab_loc_p);
            int16x4_t _tab0    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 0) * 4);
            int16x4_t _tab1    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 1) * 4);
            int16x4_t _tab2    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 2) * 4);
            int16x4_t _tab3    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 3) * 4);

            int32x4_t _val0, _val1, _val2, _val3, _res0123;
            int16x4_t _res16;
            uint8x8x3_t _resu8;

            CAL_C0();
            CAL_C1();
            CAL_C2();

            vst3_lane_u8(dst_p, _resu8, 0);
            vst3_lane_u8(dst_p + 3, _resu8, 1);
            vst3_lane_u8(dst_p + 6, _resu8, 2);
            vst3_lane_u8(dst_p + 9, _resu8, 3);

            buf_loc_p += 4;
            tab_loc_p += 4;
            dst_p     += 12;
            ++simd_loop;
        }
        x = begin_x + (simd_loop << 2);
#endif
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
#ifdef TNN_USE_NEON
        uint8_t* dst_p         = dst +  dst_loc_base + begin_x * 4;
        int32x4_t _offset      = vdupq_n_s32(1 << 14);
        int simd_loop          = 0;
        for (; x <= end_x - 3; x += 4) {
            int32x4_t _src_loc = vld1q_s32(buf_loc_p);
            uint8x8x4_t _src01;
            _src01 = vld4_lane_u8(src1 + vgetq_lane_s32(_src_loc, 0), _src01, 0);
            _src01 = vld4_lane_u8(src1 + vgetq_lane_s32(_src_loc, 0) + 4, _src01, 1);
            _src01 = vld4_lane_u8(src2 + vgetq_lane_s32(_src_loc, 0), _src01, 2);
            _src01 = vld4_lane_u8(src2 + vgetq_lane_s32(_src_loc, 0) + 4, _src01, 3);
            _src01 = vld4_lane_u8(src1 + vgetq_lane_s32(_src_loc, 1), _src01, 4);
            _src01 = vld4_lane_u8(src1 + vgetq_lane_s32(_src_loc, 1) + 4, _src01, 5);
            _src01 = vld4_lane_u8(src2 + vgetq_lane_s32(_src_loc, 1), _src01, 6);
            _src01 = vld4_lane_u8(src2 + vgetq_lane_s32(_src_loc, 1) + 4, _src01, 7);
            int16x8_t _src16_00 = vreinterpretq_s16_u16(vmovl_u8(_src01.val[0]));
            int16x8_t _src16_01 = vreinterpretq_s16_u16(vmovl_u8(_src01.val[1]));
            int16x8_t _src16_02 = vreinterpretq_s16_u16(vmovl_u8(_src01.val[2]));
            int16x8_t _src16_03 = vreinterpretq_s16_u16(vmovl_u8(_src01.val[3]));
            uint8x8x4_t _src23;
            _src23 = vld4_lane_u8(src1 + vgetq_lane_s32(_src_loc, 2), _src23, 0);
            _src23 = vld4_lane_u8(src1 + vgetq_lane_s32(_src_loc, 2) + 4, _src23, 1);
            _src23 = vld4_lane_u8(src2 + vgetq_lane_s32(_src_loc, 2), _src23, 2);
            _src23 = vld4_lane_u8(src2 + vgetq_lane_s32(_src_loc, 2) + 4, _src23, 3);
            _src23 = vld4_lane_u8(src1 + vgetq_lane_s32(_src_loc, 3), _src23, 4);
            _src23 = vld4_lane_u8(src1 + vgetq_lane_s32(_src_loc, 3) + 4, _src23, 5);
            _src23 = vld4_lane_u8(src2 + vgetq_lane_s32(_src_loc, 3), _src23, 6);
            _src23 = vld4_lane_u8(src2 + vgetq_lane_s32(_src_loc, 3) + 4, _src23, 7);
            int16x8_t _src16_10 = vreinterpretq_s16_u16(vmovl_u8(_src23.val[0]));
            int16x8_t _src16_11 = vreinterpretq_s16_u16(vmovl_u8(_src23.val[1]));
            int16x8_t _src16_12 = vreinterpretq_s16_u16(vmovl_u8(_src23.val[2]));
            int16x8_t _src16_13 = vreinterpretq_s16_u16(vmovl_u8(_src23.val[3]));

            int16x4_t _tab_loc = vld1_s16(tab_loc_p);
            int16x4_t _tab0    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 0) * 4);
            int16x4_t _tab1    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 1) * 4);
            int16x4_t _tab2    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 2) * 4);
            int16x4_t _tab3    = vld1_s16(tab_p + vget_lane_s16(_tab_loc, 3) * 4);

            int32x4_t _val0, _val1, _val2, _val3, _res0123;
            int16x4_t _res16;
            uint8x8x4_t _resu8;

            CAL_C0();
            CAL_C1();
            CAL_C2();
            CAL_C3();

            vst4_lane_u8(dst_p, _resu8, 0);
            vst4_lane_u8(dst_p + 4, _resu8, 1);
            vst4_lane_u8(dst_p + 8, _resu8, 2);
            vst4_lane_u8(dst_p + 12, _resu8, 3);

            buf_loc_p += 4;
            tab_loc_p += 4;
            dst_p     += 16;
            ++simd_loop;
        }
        x = begin_x + (simd_loop << 2);
#endif
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

#ifdef TNN_USE_NEON

#undef MAKE_CAL
#undef CAL_C0
#undef CAL_C1
#undef CAL_C2
#undef CAL_C3

#endif

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

        WarpAffinePrepareOneRow(buf_loc_t, tab_loc_t, adelta, bdelta, schannel, src, src_w, src_h,
                                dst + dst_loc_base, dst_w, y % dst_h, (y / dst_h) * src_plane, x_count, end_x, border_val);
        WarpAffineCalculateOneRow(end_x - x_count + 1, end_x, schannel, dst_loc_base, buf_loc_t, tab_loc_t, src, src2, dst);
    }

    delete[] buf_loc;
    delete[] tab_loc;

    free(buffer);
}

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
        const uint8_t* srcY  = src + b * src_plane;
        uint8_t* dstY        = dst + b * dst_plane;
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
    int* buffer = nullptr;
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

            bool is_left  = ((new_x >> 5) & 31) < 16;
            bool is_top   = ((new_y >> 5) & 31) < 16;

            int src_loc   = (new_x_loc + new_y_loc * src_w) * schannel;
            auto src_y1   = src_b + src_loc;
            auto src_y2   = src_y1 + src_stride;
            auto dst_x    = dst_y + x * schannel;

            if (CheckDataIsInBoundary(new_x_loc, new_y_loc, src_w, src_h)) {
                int c = 0;
#ifdef TNN_USE_NEON
                if (schannel == 4) {
                    uint8x8_t v_is_top = is_top ? vdup_n_u8(255) : vdup_n_u8(0);
                    uint8x8_t v_src_1  = vld1_u8(src_y1);
                    uint8x8_t v_src_2  = vld1_u8(src_y2);
                    uint8x8_t v_src_h  = vbsl_u8(v_is_top, v_src_1, v_src_2);
                    if (is_left) {
                        vst1_lane_u8(dst_x, v_src_h, 0);
                        vst1_lane_u8(dst_x + 1, v_src_h, 1);
                        vst1_lane_u8(dst_x + 2, v_src_h, 2);
                        vst1_lane_u8(dst_x + 3, v_src_h, 3);
                    } else {
                        vst1_lane_u8(dst_x, v_src_h, 4);
                        vst1_lane_u8(dst_x + 1, v_src_h, 5);
                        vst1_lane_u8(dst_x + 2, v_src_h, 6);
                        vst1_lane_u8(dst_x + 3, v_src_h, 7);
                    }
                    c = 4;
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
        const uint8_t* srcY  = src + b * src_plane;
        uint8_t* dstY        = dst + b * dst_plane;
        WarpAffineNearestC1(srcY, 1, src_w, src_h, dstY, dst_w, dst_h, transform, border_val);

        const uint8_t* srcUV = srcY + src_w * src_h;
        uint8_t* dstUV       = dstY + dst_w * dst_h;
        WarpAffineNearestC2(srcUV, 1, src_w / 2, src_h / 2, dstUV, dst_w / 2, dst_h / 2, transform, border_val);
    }
}

}  // namespace TNN_NS
