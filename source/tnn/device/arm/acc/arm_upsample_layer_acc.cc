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

#include "tnn/device/arm/acc/arm_upsample_layer_acc.h"

#include "math.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

#define SATURATE_CAST_SHORT(X)                                                                                         \
    (short)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX)

static inline bool need_do_scale(const float *scale, int len) {
    for (int i = 0; i < len; ++i) {
        if (fabs(scale[i] - 1.0) > 0.0078125) {
            return true;
        }
    }
    return false;
}

static inline int upsample_nearest2d(float *output_data, const float *input_data, int ih, int iw, int oh, int ow,
                                     int c_4) {
    auto src_z_step = iw * ih * 4;
    auto dst_z_step = ow * oh * 4;

    const float height_scale = (float)ih / (float)oh;
    const float width_scale  = (float)iw / (float)ow;

    OMP_PARALLEL_FOR_
    for (int z = 0; z < c_4; z++) {
        auto dst_z = output_data + z * dst_z_step;
        auto src_z = input_data + z * src_z_step;
        for (int h = 0; h < oh; h++) {
            int scale_h = h * height_scale;
            auto dst_y  = dst_z + h * ow * 4;
            auto src_y  = src_z + scale_h * iw * 4;
            for (int w = 0; w < ow; w++) {
                int scale_w = w * width_scale;
                Float4::save(dst_y + w * 4, Float4::load(src_y + scale_w * 4));
            }
        }
    }

    return 0;
}

template <bool do_scale>
static int upsample_nearest2d(int8_t *output_data, const int8_t *input_data, int ih, int iw, int oh, int ow, int c_4,
                              const float *scale) {
    auto c_r4       = c_4 * 4;
    auto src_y_step = iw * c_r4;
    auto dst_y_step = ow * c_r4;

    const float height_scale = (float)ih / (float)oh;
    const float width_scale  = (float)iw / (float)ow;

    OMP_PARALLEL_FOR_
    for (int h = 0; h < oh; h++) {
        int scale_h = h * height_scale;
        auto dst_y  = output_data + h * dst_y_step;
        auto src_y  = input_data + scale_h * src_y_step;
        for (int w = 0; w < ow; w++) {
            int scale_w = w * width_scale;
            auto dst_x  = dst_y + w * c_r4;
            auto src_x  = src_y + scale_w * c_r4;
            if (!do_scale) {
                memcpy(dst_x, src_x, c_r4);
            } else {
#ifndef TNN_USE_NEON
                for (int c = 0; c < c_r4; ++c) {
                    dst_x[c] = float2int8(src_x[c] * scale[c]);
                }
#else
                auto scale_p = scale;
                for (int z = 0; z < c_4 / 2; z++) {
                    float32x4_t v_scale0 = vld1q_f32(scale_p);
                    float32x4_t v_scale1 = vld1q_f32(scale_p + 4);
                    int8x8_t data_src    = vld1_s8(src_x);
                    int16x8_t data_h     = vmovl_s8(data_src);
                    float32x4_t data_f0  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(data_h)));
                    float32x4_t data_f1  = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data_h)));
                    float32x4_t res_f0   = vmulq_f32(data_f0, v_scale0);
                    float32x4_t res_f1   = vmulq_f32(data_f1, v_scale1);
                    int16x4_t res_s16    = vqmovn_s32(VCVTAQ_S32_F32(res_f0));
                    vst1_s8(dst_x, vqmovn_s16(VQMOVN_HIGH_S32_T(res_s16, VCVTAQ_S32_F32(res_f1))));

                    src_x += 8;
                    dst_x += 8;
                    scale_p += 8;
                }
                if (c_4 % 2) {
                    float32x4_t v_scale = vld1q_f32(scale_p);
                    int8x8_t data_src   = int8x8_t();
                    data_src            = vld1_lane_s8(src_x, data_src, 0);
                    data_src            = vld1_lane_s8(src_x + 1, data_src, 1);
                    data_src            = vld1_lane_s8(src_x + 2, data_src, 2);
                    data_src            = vld1_lane_s8(src_x + 3, data_src, 3);
                    int16x8_t data_h    = vmovl_s8(data_src);
                    float32x4_t data_f  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(data_h)));
                    float32x4_t res_f   = vmulq_f32(data_f, v_scale);
                    int16x4_t res_s16   = vqmovn_s32(VCVTAQ_S32_F32(res_f));
                    int8x8_t res_s8     = vqmovn_s16(vcombine_s16(res_s16, res_s16));
                    vst1_lane_s32(reinterpret_cast<int32_t *>(dst_x), vreinterpret_s32_s8(res_s8), 0);
                }
#endif
            }
        }
    }

    return 0;
}

static inline void get_bilinear_coeffs(float *h_coeffs_ptr, float *w_coeffs_ptr, int ih, int iw, int oh, int ow,
                                       bool align_corners) {
    if (align_corners) {
        const float rheight = (oh > 1) ? (float)(ih - 1) / (oh - 1) : 0.f;
        const float rwidth  = (ow > 1) ? (float)(iw - 1) / (ow - 1) : 0.f;
        for (int h = 0; h < oh; ++h) {
            h_coeffs_ptr[h] = h * rheight;
        }
        for (int w = 0; w < ow; ++w) {
            w_coeffs_ptr[w] = w * rwidth;
        }
    } else {
        const float rheight = (oh > 1) ? (float)(ih) / (oh) : 0.f;
        const float rwidth  = (ow > 1) ? (float)(iw) / (ow) : 0.f;
        for (int h = 0; h < oh; ++h) {
            h_coeffs_ptr[h] = rheight * (h + 0.5) - 0.5;
            h_coeffs_ptr[h] = h_coeffs_ptr[h] >= 0 ? h_coeffs_ptr[h] : 0;
        }
        for (int w = 0; w < ow; ++w) {
            w_coeffs_ptr[w] = rwidth * (w + 0.5) - 0.5;
            w_coeffs_ptr[w] = w_coeffs_ptr[w] >= 0 ? w_coeffs_ptr[w] : 0;
        }
    }
}

static inline int upsample_bilinear2d(float *output_data, const float *input_data, int batch, int ih, int iw, int oh,
                                      int ow, int c_4, bool align_corners) {
    auto src_z_step = iw * ih * 4;
    auto dst_z_step = ow * oh * 4;
    auto src_y_step = iw * 4;
    auto src_plane  = iw * ih * c_4 * 4;
    auto dst_plane  = ow * oh * c_4 * 4;

    RawBuffer h_coeffs(oh * sizeof(float));
    RawBuffer w_coeffs(ow * sizeof(float));
    auto h_coeffs_ptr = h_coeffs.force_to<float *>();
    auto w_coeffs_ptr = w_coeffs.force_to<float *>();

    get_bilinear_coeffs(h_coeffs_ptr, w_coeffs_ptr, ih, iw, oh, ow, align_corners);

    for (int b = 0; b < batch; ++b) {
        auto input_b  = input_data + b * src_plane;
        auto output_b = output_data + b * dst_plane;

        OMP_PARALLEL_FOR_
        for (int h2 = 0; h2 < oh; ++h2) {
            const float h1r      = h_coeffs_ptr[h2];
            const int h1         = h1r;
            const int h1p        = (h1 < ih - 1) ? 1 : 0;
            const float h1lambda = h1r - h1;
            const float h0lambda = (float)1. - h1lambda;
            for (int w2 = 0; w2 < ow; ++w2) {
                const float w1r      = w_coeffs_ptr[w2];
                const int w1         = w1r;
                const int w1p        = (w1 < iw - 1) ? 1 : 0;
                const float w1lambda = w1r - w1;
                const float w0lambda = (float)1. - w1lambda;
                const float *Xdata   = &(input_b[h1 * iw * 4 + w1 * 4]);
                float *Ydata         = &(output_b[h2 * ow * 4 + w2 * 4]);
                for (int z = 0; z < c_4; z++) {
                    Float4::save(
                        Ydata, (Float4::load(Xdata) * w0lambda + Float4::load(Xdata + w1p * 4) * w1lambda) * h0lambda +
                                   (Float4::load(Xdata + h1p * src_y_step) * w0lambda +
                                    Float4::load(Xdata + h1p * src_y_step + w1p * 4) * w1lambda) *
                                       h1lambda);

                    Xdata += src_z_step;
                    Ydata += dst_z_step;
                }
            }
        }
    }

    return 0;
}

static void calculate_position_ratio(int length, double scale, int border, int channel, int *position, short *ratio,
                                     bool align_corners) {
    const int INTER_RESIZE_COEF_BITS  = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    for (int i = 0; i < length; i++) {
        float rat_f = align_corners ? (float)(i * scale) : (float)((i + 0.5) * scale - 0.5);
        int pos_i   = static_cast<int>(floor(rat_f));
        rat_f       = rat_f - pos_i;
        if (pos_i < 0) {
            pos_i = 0;
            rat_f = 0.f;
        }
        if (pos_i >= border - 1) {
            pos_i = border - 2;
            rat_f = 1.f;
        }
        position[i] = pos_i * channel;

        float a0         = (1.f - rat_f) * INTER_RESIZE_COEF_SCALE;
        float a1         = rat_f * INTER_RESIZE_COEF_SCALE;
        ratio[i * 2]     = SATURATE_CAST_SHORT(a0);
        ratio[i * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }
}

static void get_upsample_buf(int src_w, int src_h, int w, int h, int c, int **buf, bool align_corners) {
    double scale_x;
    double scale_y;
    if (align_corners) {
        scale_x = (w > 1) ? (double)(src_w - 1) / (w - 1) : 0.0;
        scale_y = (h > 1) ? (double)(src_h - 1) / (h - 1) : 0.0;
    } else {
        scale_x = (w > 1) ? (double)src_w / w : 0.0;
        scale_y = (h > 1) ? (double)src_h / h : 0.0;
    }
    *buf          = new int[w + h + w + h];
    int *xofs     = *buf;
    int *yofs     = *buf + w;
    short *ialpha = (short *)(*buf + w + h);
    short *ibeta  = (short *)(*buf + w + h + w);

    calculate_position_ratio(w, scale_x, src_w, c, xofs, ialpha, align_corners);
    calculate_position_ratio(h, scale_y, src_h, 1, yofs, ibeta, align_corners);
}

struct UpsampleBilinearKernelParm {
    UpsampleBilinearKernelParm(int *_xofs, int *_yofs, short *_ialpha, short *_ibeta, const int8_t *_src, int8_t *_dst,
                               int _src_plane, int _src_stride) {
        xofs       = _xofs;
        yofs       = _yofs;
        ialpha     = _ialpha;
        ibeta      = _ibeta;
        src        = _src;
        dst        = _dst;
        src_plane  = _src_plane;
        src_stride = _src_stride;
    };

    int *xofs;
    int *yofs;
    short *ialpha;
    short *ibeta;
    const int8_t *src;
    int8_t *dst;
    int src_plane;
    int src_stride;
};

static void upsample_get_adjacent_rows(int sy, int prev_sy, short **rows0, short **rows1, int *xofs, const int8_t *src,
                                       int src_stride, int w, const short *ialphap) {
    if (sy == prev_sy) {
        // reuse all rows
    } else if (sy == prev_sy + 1) {
        // hresize one row
        short *rows0_old = *rows0;
        *rows0           = *rows1;
        *rows1           = rows0_old;
        const int8_t *S1 = src + src_stride * (sy + 1);

        short *rows1p = *rows1;
        for (int dx = 0; dx < w; dx++) {
            int sx   = xofs[dx];
            short a0 = ialphap[0];
            short a1 = ialphap[1];

            const int8_t *S1p = S1 + sx;

#ifndef TNN_USE_NEON
            for (int dc = 0; dc < 4; ++dc) {
                rows1p[dc] = (S1p[dc] * a0 + S1p[dc + 4] * a1) >> 4;
            }
#else
            int16x4_t _a0 = vdup_n_s16(a0);
            int16x4_t _a1 = vdup_n_s16(a1);
            int8x8_t _S1 = vld1_s8(S1p);

            int16x8_t _S116 = vmovl_s8(_S1);
            int16x4_t _S1low = vget_low_s16(_S116);
            int16x4_t _S1high = vget_high_s16(_S116);
            int32x4_t _rows1 = vmull_s16(_S1low, _a0);
            _rows1 = vmlal_s16(_rows1, _S1high, _a1);
            int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
            vst1_s16(rows1p, _rows1_sr4);
#endif
            ialphap += 2;
            rows1p += 4;
        }
    } else {
        // hresize two rows
        const int8_t *S0 = src + src_stride * (sy);
        const int8_t *S1 = src + src_stride * (sy + 1);

        short *rows0p = *rows0;
        short *rows1p = *rows1;
        for (int dx = 0; dx < w; dx++) {
            int sx   = xofs[dx];
            short a0 = ialphap[0];
            short a1 = ialphap[1];

            const int8_t *S0p = S0 + sx;
            const int8_t *S1p = S1 + sx;

#ifndef TNN_USE_NEON
            for (int dc = 0; dc < 4; ++dc) {
                rows0p[dc] = (S0p[dc] * a0 + S0p[dc + 4] * a1) >> 4;
                rows1p[dc] = (S1p[dc] * a0 + S1p[dc + 4] * a1) >> 4;
            }
#else
            int16x4_t _a0 = vdup_n_s16(a0);
            int16x4_t _a1 = vdup_n_s16(a1);
            int8x8_t _S0 = vld1_s8(S0p);
            int8x8_t _S1 = vld1_s8(S1p);
            int16x8_t _S016 = vmovl_s8(_S0);
            int16x8_t _S116 = vmovl_s8(_S1);
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
#endif
            ialphap += 2;
            rows0p += 4;
            rows1p += 4;
        }
    }
}

static void upsample_calculate_one_row(short *rows0p, short *rows1p, const int b0, const int b1, const int w,
                                       int8_t *Dp) {
#ifndef TNN_USE_NEON
    int remain = w * 4;
#else
    int nn = (w * 4) >> 3;
    int remain = (w * 4) - (nn << 3);
    int16x4_t _b0 = vdup_n_s16(b0);
    int16x4_t _b1 = vdup_n_s16(b1);
    int32x4_t _v2 = vdupq_n_s32(2);
    for (; nn > 0; nn--) {
        int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
        int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
        int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p + 4);
        int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p + 4);

        int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
        int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
        int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
        int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

        int32x4_t _acc = _v2;
        _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
        _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

        int32x4_t _acc_1 = _v2;
        _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
        _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

        int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
        int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

        int8x8_t _D = vqmovn_s16(vcombine_s16(_acc16, _acc16_1));

        vst1_s8(Dp, _D);

        Dp += 8;
        rows0p += 8;
        rows1p += 8;
    }
#endif
    for (; remain; --remain) {
        *Dp++ =
            (int8_t)(((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
    }
}

void upsample_bilinear_one_row(UpsampleBilinearKernelParm &param, int thread_id, short **rows0_t, short **rows1_t,
                               int *prev_sy, int b, int w, int h, int stride, int dy) {
    int sy = param.yofs[dy];
    upsample_get_adjacent_rows(sy, prev_sy[thread_id], &rows0_t[thread_id], &rows1_t[thread_id], param.xofs,
                               param.src + b * param.src_plane, param.src_stride, w, param.ialpha);
    prev_sy[thread_id] = sy;

    // vresize
    short b0 = param.ibeta[dy * 2];
    short b1 = param.ibeta[dy * 2 + 1];

    int8_t *Dp = param.dst + stride * (b * h + dy);

    upsample_calculate_one_row(rows0_t[thread_id], rows1_t[thread_id], b0, b1, w, Dp);
}

static int upsample_bilinear_c4(int8_t *dst, const int8_t *src, int batch, int src_h, int src_w, int h, int w,
                                bool align_corners) {
    int src_stride = src_w * 4;
    int stride     = w * 4;
    int *buf       = nullptr;
    get_upsample_buf(src_w, src_h, w, h, 4, &buf, align_corners);
    int *xofs     = buf;
    int *yofs     = buf + w;
    short *ialpha = (short *)(buf + w + h);
    short *ibeta  = (short *)(buf + w + h + w);
    int src_plane = src_h * src_stride;

    UpsampleBilinearKernelParm param(xofs, yofs, ialpha, ibeta, src, dst, src_plane, src_stride);

    // loop body
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    short *rows0        = new short[(w * 4) * max_num_threads];
    short *rows1        = new short[(w * 4) * max_num_threads];
    short *rows0_t[max_num_threads];
    short *rows1_t[max_num_threads];
    int prev_sy[max_num_threads];

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < max_num_threads; ++t) {
            prev_sy[t] = -2;
            rows0_t[t] = rows0 + t * (w * 4);
            rows1_t[t] = rows1 + t * (w * 4);
        }

        OMP_PARALLEL_FOR_
        for (int dy = 0; dy < h; dy++) {
            int thread_id = OMP_TID_;
            upsample_bilinear_one_row(param, thread_id, rows0_t, rows1_t, prev_sy, b, w, h, stride, dy);
        }
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
    return 0;
}

template <bool do_scale>
static void upsample_bilinear_cn(int8_t *output_data, const int8_t *input_data, const float *h_coeffs_ptr,
                                 const float *w_coeffs_ptr, int c_4, int ih, int iw, int oh, int ow,
                                 const float *scale) {
    auto c_r4       = c_4 * 4;
    auto src_y_step = iw * c_r4;
    auto dst_y_step = ow * c_r4;

    const float INTER_RESIZE_COEF_SCALE = float(1 << 11);

    OMP_PARALLEL_FOR_
    for (int h2 = 0; h2 < oh; ++h2) {
        const float h1r      = h_coeffs_ptr[h2];
        const int h1         = h1r;
        const int h1p        = (h1 < ih - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = (float)1. - h1lambda;
        const short h1_short = SATURATE_CAST_SHORT(h1lambda * INTER_RESIZE_COEF_SCALE);
        const short h0_short = SATURATE_CAST_SHORT(h0lambda * INTER_RESIZE_COEF_SCALE);
        for (int w2 = 0; w2 < ow; ++w2) {
            const float w1r       = w_coeffs_ptr[w2];
            const int w1          = w1r;
            const int w1p         = (w1 < iw - 1) ? 1 : 0;
            const float w1lambda  = w1r - w1;
            const float w0lambda  = (float)1. - w1lambda;
            const short w1_short  = SATURATE_CAST_SHORT(w1lambda * INTER_RESIZE_COEF_SCALE);
            const short w0_short  = SATURATE_CAST_SHORT(w0lambda * INTER_RESIZE_COEF_SCALE);
            const int8_t *Xdata00 = &(input_data[h1 * src_y_step + w1 * c_r4]);
            const int8_t *Xdata01 = Xdata00 + w1p * c_r4;
            const int8_t *Xdata10 = Xdata00 + h1p * src_y_step;
            const int8_t *Xdata11 = Xdata10 + w1p * c_r4;
            int8_t *Ydata         = &(output_data[h2 * dst_y_step + w2 * c_r4]);
            const float *scale_p  = scale;
#ifndef TNN_USE_NEON
            for (int c = 0; c < c_r4; ++c) {
                if (do_scale) {
                    // compute as float
                    Ydata[c] = float2int8(((Xdata00[c] * w0lambda + Xdata01[c] * w1lambda) * h0lambda +
                                           (Xdata10[c] * w0lambda + Xdata11[c] * w1lambda) * h1lambda) *
                                          scale_p[c]);
                } else {
                    // compute as int
                    short h0_res = (Xdata00[c] * w0_short + Xdata01[c] * w1_short) >> 4;
                    short h1_res = (Xdata10[c] * w0_short + Xdata11[c] * w1_short) >> 4;
                    int8_t res   = (((h0_res * h0_short) >> 16) + ((h1_res * h1_short) >> 16) + 2) >> 2;
                    Ydata[c]     = res;
                }
            }
#else
            if (do_scale) {
                float32x4_t v_w0lambda = vdupq_n_f32(w0lambda);
                float32x4_t v_w1lambda = vdupq_n_f32(w1lambda);
                float32x4_t v_h0lambda = vdupq_n_f32(h0lambda);
                float32x4_t v_h1lambda = vdupq_n_f32(h1lambda);
                for (int z = 0; z < c_4 / 2; z++) {
                    float32x4_t v_scale0 = vld1q_f32(scale_p);
                    float32x4_t v_scale1 = vld1q_f32(scale_p + 4);
                    int8x8_t data00 = vld1_s8(Xdata00);
                    int8x8_t data01 = vld1_s8(Xdata01);
                    int8x8_t data10 = vld1_s8(Xdata10);
                    int8x8_t data11 = vld1_s8(Xdata11);
                    int16x8_t data00h = vmovl_s8(data00);
                    int16x8_t data01h = vmovl_s8(data01);
                    int16x8_t data10h = vmovl_s8(data10);
                    int16x8_t data11h = vmovl_s8(data11);
                    float32x4_t data00_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(data00h)));
                    float32x4_t data00_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data00h)));
                    float32x4_t data01_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(data01h)));
                    float32x4_t data01_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data01h)));
                    float32x4_t data10_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(data10h)));
                    float32x4_t data10_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data10h)));
                    float32x4_t data11_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(data11h)));
                    float32x4_t data11_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data11h)));
                    float32x4_t acc0 = vmlaq_f32(vmulq_f32(data00_0, v_w0lambda), data01_0, v_w1lambda);
                    float32x4_t acc1 = vmlaq_f32(vmulq_f32(data10_0, v_w0lambda), data11_0, v_w1lambda);
                    float32x4_t acc = vmulq_f32(vmlaq_f32(vmulq_f32(acc0, v_h0lambda), acc1, v_h1lambda), v_scale0);
                    int16x4_t res_s16 = vqmovn_s32(VCVTAQ_S32_F32(acc));
                    acc0 = vmlaq_f32(vmulq_f32(data00_1, v_w0lambda), data01_1, v_w1lambda);
                    acc1 = vmlaq_f32(vmulq_f32(data10_1, v_w0lambda), data11_1, v_w1lambda);
                    acc = vmulq_f32(vmlaq_f32(vmulq_f32(acc0, v_h0lambda), acc1, v_h1lambda), v_scale1);
                    vst1_s8(Ydata, vqmovn_s16(VQMOVN_HIGH_S32_T(res_s16, VCVTAQ_S32_F32(acc))));

                    Xdata00 += 8;
                    Xdata01 += 8;
                    Xdata10 += 8;
                    Xdata11 += 8;
                    Ydata += 8;
                    scale_p += 8;
                }
                if (c_4 % 2) {
                    float32x4_t v_scale = vld1q_f32(scale_p);
                    int8x8_t data00 = vld1_s8(Xdata00);
                    int8x8_t data01 = vld1_s8(Xdata01 - 4);
                    int8x8_t data10 = vld1_s8(Xdata10 - 4);
                    int8x8_t data11 = vld1_s8(Xdata11 - 4);
                    int16x8_t data00h = vmovl_s8(data00);
                    int16x8_t data01h = vmovl_s8(data01);
                    int16x8_t data10h = vmovl_s8(data10);
                    int16x8_t data11h = vmovl_s8(data11);
                    float32x4_t data00_1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(data00h)));
                    float32x4_t data01_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data01h)));
                    float32x4_t data10_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data10h)));
                    float32x4_t data11_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data11h)));
                    float32x4_t acc0 = vmlaq_f32(vmulq_f32(data00_1, v_w0lambda), data01_1, v_w1lambda);
                    float32x4_t acc1 = vmlaq_f32(vmulq_f32(data10_1, v_w0lambda), data11_1, v_w1lambda);
                    float32x4_t acc = vmulq_f32(vmlaq_f32(vmulq_f32(acc0, v_h0lambda), acc1, v_h1lambda), v_scale);
                    int16x4_t res_s16 = vqmovn_s32(VCVTAQ_S32_F32(acc));
                    int8x8_t res_s8 = vqmovn_s16(vcombine_s16(res_s16, res_s16));
                    vst1_lane_s32(reinterpret_cast<int32_t *>(Ydata), vreinterpret_s32_s8(res_s8), 0);
                }
            } else {
                int16x4_t v_w0 = vdup_n_s16(w0_short);
                int16x4_t v_w1 = vdup_n_s16(w1_short);
                int16x4_t v_h0 = vdup_n_s16(h0_short);
                int16x4_t v_h1 = vdup_n_s16(h1_short);
                int32x4_t v_2 = vdupq_n_s32(2);
                for (int z = 0; z < c_4 / 2; z++) {
                    int8x8_t data00 = vld1_s8(Xdata00);
                    int8x8_t data01 = vld1_s8(Xdata01);
                    int8x8_t data10 = vld1_s8(Xdata10);
                    int8x8_t data11 = vld1_s8(Xdata11);
                    int16x8_t data00h = vmovl_s8(data00);
                    int16x8_t data01h = vmovl_s8(data01);
                    int16x8_t data10h = vmovl_s8(data10);
                    int16x8_t data11h = vmovl_s8(data11);
                    int32x4_t acc0 = vmlal_s16(vmull_s16(vget_low_s16(data00h), v_w0), vget_low_s16(data01h), v_w1);
                    int32x4_t acc1 = vmlal_s16(vmull_s16(vget_low_s16(data10h), v_w0), vget_low_s16(data11h), v_w1);
                    int32x4_t acc_h0 = vsraq_n_s32(
                        v_2, vmlal_s16(vmull_s16(vshrn_n_s32(acc0, 4), v_h0), vshrn_n_s32(acc1, 4), v_h1), 16);
                    acc0 = vmlal_s16(vmull_s16(vget_high_s16(data00h), v_w0), vget_high_s16(data01h), v_w1);
                    acc1 = vmlal_s16(vmull_s16(vget_high_s16(data10h), v_w0), vget_high_s16(data11h), v_w1);
                    int32x4_t acc_h1 = vsraq_n_s32(
                        v_2, vmlal_s16(vmull_s16(vshrn_n_s32(acc0, 4), v_h0), vshrn_n_s32(acc1, 4), v_h1), 16);
                    vst1_s8(Ydata, vqmovn_s16(vcombine_s16(vshrn_n_s32(acc_h0, 2), vshrn_n_s32(acc_h1, 2))));

                    Xdata00 += 8;
                    Xdata01 += 8;
                    Xdata10 += 8;
                    Xdata11 += 8;
                    Ydata += 8;
                }
                if (c_4 % 2) {
                    int8x8_t data00 = vld1_s8(Xdata00);
                    int8x8_t data01 = vld1_s8(Xdata01 - 4);
                    int8x8_t data10 = vld1_s8(Xdata10 - 4);
                    int8x8_t data11 = vld1_s8(Xdata11 - 4);
                    int16x8_t data00h = vmovl_s8(data00);
                    int16x8_t data01h = vmovl_s8(data01);
                    int16x8_t data10h = vmovl_s8(data10);
                    int16x8_t data11h = vmovl_s8(data11);
                    int32x4_t acc0 = vmlal_s16(vmull_s16(vget_low_s16(data00h), v_w0), vget_high_s16(data01h), v_w1);
                    int32x4_t acc1 = vmlal_s16(vmull_s16(vget_high_s16(data10h), v_w0), vget_high_s16(data11h), v_w1);
                    int32x4_t acc = vsraq_n_s32(
                        v_2, vmlal_s16(vmull_s16(vshrn_n_s32(acc0, 4), v_h0), vshrn_n_s32(acc1, 4), v_h1), 16);
                    int16x4_t res_s16 = vshrn_n_s32(acc, 2);
                    int8x8_t res_s8 = vqmovn_s16(vcombine_s16(res_s16, res_s16));
                    vst1_lane_s32(reinterpret_cast<int32_t *>(Ydata), vreinterpret_s32_s8(res_s8), 0);
                }
            }
#endif
        }
    }
}

template <bool do_scale>
static int upsample_bilinear2d(int8_t *output_data, const int8_t *input_data, int batch, int ih, int iw, int oh, int ow,
                               int c_4, bool align_corners, const float *scale) {
    if (!do_scale && c_4 == 1) {
        return upsample_bilinear_c4(output_data, input_data, batch, ih, iw, oh, ow, align_corners);
    }

    auto src_plane = iw * ih * c_4 * 4;
    auto dst_plane = ow * oh * c_4 * 4;

    RawBuffer h_coeffs(oh * sizeof(float));
    RawBuffer w_coeffs(ow * sizeof(float));
    auto h_coeffs_ptr = h_coeffs.force_to<float *>();
    auto w_coeffs_ptr = w_coeffs.force_to<float *>();

    get_bilinear_coeffs(h_coeffs_ptr, w_coeffs_ptr, ih, iw, oh, ow, align_corners);

    for (int b = 0; b < batch; ++b) {
        auto input_b  = input_data + b * src_plane;
        auto output_b = output_data + b * dst_plane;
        if (do_scale) {
            upsample_bilinear_cn<true>(output_b, input_b, h_coeffs_ptr, w_coeffs_ptr, c_4, ih, iw, oh, ow, scale);
        } else {
            upsample_bilinear_cn<false>(output_b, input_b, h_coeffs_ptr, w_coeffs_ptr, c_4, ih, iw, oh, ow, scale);
        }
    }

    return 0;
}

ArmUpsampleLayerAcc::~ArmUpsampleLayerAcc() {}

Status ArmUpsampleLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<UpsampleLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    auto dims_input   = inputs[0]->GetBlobDesc().dims;
    auto dims_output  = outputs[0]->GetBlobDesc().dims;
    auto batch        = dims_input[0];
    auto input_plane  = dims_input[2] * dims_input[3] * ROUND_UP(dims_input[1], 4);
    auto output_plane = dims_output[2] * dims_output[3] * ROUND_UP(dims_output[1], 4);

    DataType data_type = outputs[0]->GetBlobDesc().data_type;

    if (data_type == DATA_TYPE_INT8) {
        auto dims_output    = outputs[0]->GetBlobDesc().dims;
        int total_byte_size = ROUND_UP(dims_output[1], 4) * sizeof(float);

        auto input_resource  = reinterpret_cast<BlobInt8 *>(inputs[0])->GetIntResource();
        auto output_resource = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource();
        const float *i_scale = input_resource->scale_handle.force_to<float *>();
        const float *o_scale = output_resource->scale_handle.force_to<float *>();
        int scale_len_i      = input_resource->scale_handle.GetDataCount();
        int scale_len_o      = output_resource->scale_handle.GetDataCount();

        if (buffer_scale_.GetBytesSize() < total_byte_size) {
            buffer_scale_ = RawBuffer(total_byte_size);
        }
        float *temp_ptr = buffer_scale_.force_to<float *>();
        for (int i = 0; i < dims_output[1]; i++) {
            int scale_idx_i = scale_len_i == 1 ? 0 : i;
            int scale_idx_o = scale_len_o == 1 ? 0 : i;
            if (o_scale[scale_idx_o] >= FLT_MIN)
                temp_ptr[i] = i_scale[scale_idx_i] / o_scale[scale_idx_o];
            else
                temp_ptr[i] = 0.0;
        }
        do_scale_ = need_do_scale(temp_ptr, dims_output[1]);
    } else {
        do_scale_ = false;
    }

    float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

    auto oc_4 = UP_DIV(dims_output[1], 4);

    if (dims_input[2] == dims_output[2] && dims_input[3] == dims_output[3] && !do_scale_) {
        if (output_data != input_data) {
            memcpy(output_data, input_data, batch * input_plane * DataTypeUtils::GetBytesSize(data_type));
        }
    } else if (param->mode == 1) {  // nearest
        if (data_type == DATA_TYPE_FLOAT) {
            for (int b = 0; b < batch; ++b) {
                upsample_nearest2d(output_data + b * output_plane, input_data + b * input_plane, dims_input[2],
                                   dims_input[3], dims_output[2], dims_output[3], oc_4);
            }
        } else if (data_type == DATA_TYPE_INT8) {
            for (int b = 0; b < batch; ++b) {
                auto output_b = reinterpret_cast<int8_t *>(output_data) + b * output_plane;
                auto input_b  = reinterpret_cast<int8_t *>(input_data) + b * input_plane;
                if (do_scale_)
                    upsample_nearest2d<true>(output_b, input_b, dims_input[2], dims_input[3], dims_output[2],
                                             dims_output[3], oc_4, buffer_scale_.force_to<float *>());
                else
                    upsample_nearest2d<false>(output_b, input_b, dims_input[2], dims_input[3], dims_output[2],
                                              dims_output[3], oc_4, buffer_scale_.force_to<float *>());
            }
        } else {
            return Status(TNNERR_LAYER_ERR, "Error: Not supported data type for upsample nearest");
        }
    } else if (param->mode == 2) {  // bilinear/linear
        if (data_type == DATA_TYPE_FLOAT) {
            upsample_bilinear2d(output_data, input_data, batch, dims_input[2], dims_input[3], dims_output[2],
                                dims_output[3], oc_4, (bool)param->align_corners);
        } else if (data_type == DATA_TYPE_INT8) {
            if (do_scale_)
                upsample_bilinear2d<true>(reinterpret_cast<int8_t *>(output_data),
                                          reinterpret_cast<int8_t *>(input_data), batch, dims_input[2], dims_input[3],
                                          dims_output[2], dims_output[3], oc_4, (bool)param->align_corners,
                                          buffer_scale_.force_to<float *>());
            else
                upsample_bilinear2d<false>(reinterpret_cast<int8_t *>(output_data),
                                           reinterpret_cast<int8_t *>(input_data), batch, dims_input[2], dims_input[3],
                                           dims_output[2], dims_output[3], oc_4, (bool)param->align_corners,
                                           buffer_scale_.force_to<float *>());
        } else {
            return Status(TNNERR_LAYER_ERR, "Error: Not supported data type for upsample bilinear");
        }
    } else {
        LOGE("Error: Upsample dont support resize mode\n");
        return Status(TNNERR_MODEL_ERR, "Error: Upsample dont support resize mode");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Upsample, LAYER_UPSAMPLE)

}  // namespace TNN_NS
