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

#include "tnn/device/arm/acc/compute/compute.h"

#include <string.h>

#include "tnn/core/macro.h"
#include "tnn/device/arm/acc/Float4.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_util.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

#ifndef TNN_USE_NEON
void GemmInt8UnitN8Naive(long mr, long nr, long k, const int8_t* a, long a_stride, const void* w, int8_t* c,
                         long c_stride, const float* scales, long relu, const int8_t* add_input,
                         const float* add_scale) {
    union {
        const void* as_void_ptr;
        int8_t* as_int8_ptr;
        int32_t* as_int32_ptr;
    } packed = {w};

    for (int m = 0; m < mr; m++) {
        for (int n = 0; n < nr; n++) {
            int acc          = packed.as_int32_ptr[n];
            int8_t* packed_w = reinterpret_cast<int8_t*>(packed.as_int32_ptr + 8);
            for (int kk = 0; kk < k; kk++) {
                acc += (int32_t)a[m * a_stride + kk] * (int32_t)packed_w[kk * 8 + n];
            }
            auto res = acc * scales[n];
            // Conv-Relu-Add
            if (relu < 0) {
                res = MAX(0, res);
            }
            if (add_input) {
                res += add_input[m * c_stride + n] * add_scale[n];
            }
            // Conv-Add-Relu
            if (relu > 0) {
                res = MAX(0, res);
            }
            c[m * c_stride + n] = float2int8(res);
        }
    }
}
#else
extern "C" {
void GemmInt8Unit4x8(long mr, long nr, long k, const int8_t* a, long a_stride, const void* w, int8_t* c, long c_stride,
                     const float* scales, long, const int8_t* add_input, const float* add_scale);
void GemmInt8Unit8x8(long mr, long nr, long k, const int8_t* a, long a_stride, const void* w, int8_t* c, long c_stride,
                     const float* scales, long, const int8_t* add_input, const float* add_scale);
}
#endif

static void ComputeQ8GemmTile(const Q8GemmContext* context, long mr_block_start, long nr_block_start,
                              long mr_block_size, long nr_block_size) {
    const long k         = context->k;
    const long k_stride  = context->k_stride;
    const long n         = context->n;
    const long n_stride  = context->n_stride;
    const int8_t* a      = context->a;
    const long a_stride  = context->a_stride;
    const void* packed_w = context->packed_w;
    int8_t* c            = context->c;
    const long c_stride  = context->c_stride;

#ifndef TNN_USE_NEON
    GemmInt8N8Func gemm_int8_func = GemmInt8UnitN8Naive;
#elif defined(__aarch64__)
    GemmInt8N8Func gemm_int8_func = GemmInt8Unit8x8;
#else
    GemmInt8N8Func gemm_int8_func = GemmInt8Unit4x8;
#endif

    auto add_input = context->add_input ? context->add_input + mr_block_start * c_stride + nr_block_start : nullptr;
    auto add_scale = context->add_scale ? context->add_scale + nr_block_start : nullptr;

    gemm_int8_func(mr_block_size, nr_block_size, k, a + (mr_block_start)*a_stride, a_stride,
                   (const void*)((intptr_t)packed_w + nr_block_start * (k_stride * sizeof(int8_t) + sizeof(int32_t))),
                   c + mr_block_start * c_stride + nr_block_start, c_stride, context->scales + nr_block_start,
                   context->relu, add_input, add_scale);
}

void ComputeQ8Gemm(const Q8GemmContext* context, int32_t range_k, int32_t range_l, int32_t tile_k, int32_t tile_l) {
    OMP_PARALLEL_FOR_GUIDED_
    for (int32_t k = 0; k < range_k; k += tile_k) {
        for (int32_t l = 0; l < range_l; l += tile_l) {
            ComputeQ8GemmTile(context, k, l, std::min(range_k - k, tile_k), std::min(range_l - l, tile_l));
        }
    }
}

#ifndef TNN_USE_NEON
/*
kernel func used in linux debug mode
conv int8 fuse with add common micro kernel
*/
void GemmInt8Unit4x4(const int8_t* src, const int8_t* weight, int8_t* dst, long src_w_step, long dst_depth, long cdiv8,
                     const float* scale, const int32_t* bias, long relu, const int8_t* add_input,
                     const float* add_scale) {
    for (long w = 0; w < 4; ++w) {
        const auto src_x   = src + w * src_w_step;
        auto dst_x         = dst + w * dst_depth;
        auto add_input_x   = add_input ? add_input + w * dst_depth : nullptr;
        int32_t dstTemp[4] = {0, 0, 0, 0};
        long sz            = 0;
        for (; sz < cdiv8 / 2; ++sz) {
            const auto weight_sz = weight + (4 * 16) * sz;
            const auto src_z     = src_x + sz * 16;

            for (long j = 0; j < 4; ++j) {
                const auto weight_j = weight_sz + j * 16;
                for (long i = 0; i < 16; ++i) {
                    dstTemp[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
                }
            }
        }
        for (; sz < cdiv8 / 2 + cdiv8 % 2; ++sz) {
            const auto weight_sz = weight + (4 * 16) * sz;
            const auto src_z     = src_x + sz * 16;

            for (long j = 0; j < 4; ++j) {
                const auto weight_j = weight_sz + j * 16;
                for (long i = 0; i < 8; ++i) {
                    dstTemp[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
                }
            }
        }
        for (long j = 0; j < 4; ++j) {
            auto res = static_cast<float>(dstTemp[j] + bias[j]) * scale[j];
            // Conv-Relu-Add
            if (relu < 0) {
                res = MAX(0, res);
            }
            if (add_input_x) {
                res += add_input_x[j] * add_scale[j];
            }
            // Conv-Add-Relu
            if (relu > 0) {
                res = MAX(0, res);
            }
            dst_x[j] = float2int8(res);
        }
    }
}
#endif

#ifdef TNN_USE_NEON
/*
convert float data to s16, vqmovn_high_s32 can only used int armv8
(int32)(v * scale) -> int16
*/
inline int8x8_t Float4x2ScaleTos8(const float32x4_t v0, const float32x4_t v1, const float32x4_t s0,
                                  const float32x4_t s1) {
    float32x4_t mul0 = vmulq_f32(v0, s0);
    float32x4_t mul1 = vmulq_f32(v1, s1);
    int16x8_t s16    = VQMOVN_HIGH_S32_T(vqmovn_s32(VCVTAQ_S32_F32(mul0)), VCVTAQ_S32_F32(mul1));
    return vqmovn_s16(s16);
}
/*
convert float data to s8, pack four int8 values to one int32 value
(int32)(v * scale) -> int16 -> int8, pack four int8 to one int32 value
*/
inline int32_t Float4ScaleTos8(const float32x4_t v, const float32x4_t s) {
    float32x4_t mul = vmulq_f32(v, s);
    int8x8_t s8     = vqmovn_s16(vcombine_s16(vqmovn_s32(VCVTAQ_S32_F32(mul)), vdup_n_s16(0)));
    return vreinterpret_s32_s8(s8)[0];
}

/*
quant data from float to int8
*/
void FloatToInt8C4(int8_t* dst, const float* src, const float* scale, long batch, long channel, long hw) {
    const long c_4         = 4;
    float32x4_t scale_neon = vld1q_f32(scale);
    for (long n = 0; n < batch; n++) {
        int8_t* dst_c      = dst + n * c_4 * hw;
        const float* src_c = src + n * c_4 * hw;
        long idx           = hw - hw % 2;
        OMP_PARALLEL_FOR_GUIDED_
        for (long cnt = 0; cnt < idx; cnt += 2) {
            // nhwc4 to nchw4
            float32x4_t val0 = vmulq_f32(vld1q_f32(src_c + cnt * c_4), scale_neon);
            float32x4_t val1 = vmulq_f32(vld1q_f32(src_c + cnt * c_4 + 4), scale_neon);
            int16x4_t s16_0  = vqmovn_s32(VCVTAQ_S32_F32(val0));
            int16x8_t s16    = VQMOVN_HIGH_S32_T(s16_0, VCVTAQ_S32_F32(val1));
            vst1_s8(dst_c + cnt * c_4, vqmovn_s16(s16));
        }
        if (idx == hw - 1) {
            float32x4_t val0 = vmulq_f32(vld1q_f32(src_c + idx * c_4), scale_neon);
            int16x4_t s16_0  = vqmovn_s32(VCVTAQ_S32_F32(val0));
            int8x8_t s8      = vqmovn_s16(VQMOVN_HIGH_S32_T(s16_0, VCVTAQ_S32_F32(val0)));
            vst1_lane_s32((int32_t*)(dst_c + idx * c_4), vreinterpret_s32_s8(s8), 0);
        }
    }
}

/*
pack line used in gemm int8
*/
void PackLineV7(long cin, const int32_t* src, int32_t* dst) {
    cin = cin / 4;
    long temp[8];
    for (long c = 0; c < cin; c += 2) {
        int32x2x2_t v[2];
        v[0].val[0] = vld1_s32(src + c + 0 * cin);
        v[0].val[1] = vld1_s32(src + c + 1 * cin);
        v[1].val[0] = vld1_s32(src + c + 2 * cin);
        v[1].val[1] = vld1_s32(src + c + 3 * cin);
        vst2_s32(dst + c * 4, v[0]);
        vst2_s32(dst + c * 4 + 4, v[1]);
    }
}

#endif

/*
general max pooling int8 kernel
*/
void MaxPoolingINT8(const int8_t* src, long iw, long ih, int8_t* dst, long ow, long oh, long c_r4, long kw, long kh,
                    long stride_w, long stride_h, long pad_w, long pad_h) {
    OMP_PARALLEL_FOR_GUIDED_
    for (long oy = 0; oy < oh; ++oy) {
        for (long ox = 0; ox < ow; ++ox) {
            const long srcOriginX = ox * stride_w - pad_w;
            const long srcOriginY = oy * stride_h - pad_h;
            const long kxs        = MAX(0, -srcOriginX);
            const long kxe        = MIN(kw, iw - srcOriginX);
            const long kys        = MAX(0, -srcOriginY);
            const long kye        = MIN(kh, ih - srcOriginY);
            long oc               = 0;
#ifdef TNN_USE_NEON
            for (; oc < c_r4 - 4; oc += 8) {
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                int8x8_t max_reg   = vdup_n_s8(-127);
                // find kernel_w * kernel_h max value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;
                    for (; kx < kxe; kx++) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        max_reg                = vmax_s8(max_reg, vld1_s8(srcPtrStart));
                    }
                }
                vst1_s8(dst_ptr, max_reg);
            }
#endif
            for (; oc < c_r4; oc += 4) {
                int8_t maxValue[4] = {-127, -127, -127, -127};
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                // find kernel_w * kernel_h max value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;
                    for (; kx < kxe; ++kx) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        for (long j = 0; j < 4; ++j) {
                            maxValue[j] = MAX(maxValue[j], srcPtrStart[j]);
                        }
                    }
                }
                // output
                *(int32_t*)dst_ptr = *(int32_t*)maxValue;
            }
        }
    }
}

/*
general avg pooling int8 kernel
*/
void AvgPoolingINT8(const int8_t* src, long iw, long ih, int8_t* dst, long ow, long oh, long c_r4, long kw, long kh,
                    long stride_w, long stride_h, long pad_w, long pad_h) {
    for (long oy = 0; oy < oh; ++oy) {
        for (long ox = 0; ox < ow; ++ox) {
            const long srcOriginX   = ox * stride_w - pad_w;
            const long srcOriginY   = oy * stride_h - pad_h;
            const long kxs          = MAX(0, -srcOriginX);
            const long kxe          = MIN(kw, iw - srcOriginX);
            const long kys          = MAX(0, -srcOriginY);
            const long kye          = MIN(kh, ih - srcOriginY);
            const long kernel_count = (kxe - kxs) * (kye - kys);
            long oc                 = 0;
#ifdef TNN_USE_NEON
            int16_t sum[8];
            for (; oc < c_r4 - 4; oc += 8) {
                int16x8_t avg_reg  = vdupq_n_s16(0);
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                // find kernel_w * kernel_h avg value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;
                    for (; kx < kxe; kx++) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        int16x8_t cur_val      = vmovl_s8(vld1_s8(srcPtrStart));
                        avg_reg                = vaddq_s16(avg_reg, cur_val);
                    }
                }
                vst1q_s16(sum, avg_reg);
                for (long j = 0; j < 8; j++) {
                    dst_ptr[j] = sum[j] / kernel_count;
                }
            }
#endif
            for (; oc < c_r4; oc += 4) {
                int16_t sum[4]     = {0, 0, 0, 0};
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                // find kernel_w * kernel_h avg value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;

                    for (; kx < kxe; ++kx) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        for (long j = 0; j < 4; ++j) {
                            sum[j] += srcPtrStart[j];
                        }
                    }
                }
                // output
                for (long j = 0; j < 4; j++) {
                    dst_ptr[j] = static_cast<int8_t>(sum[j] / kernel_count);
                }
            }
        }
    }
}

/*
element add int8 func
*/
void MatrixAddInt8(int8_t* dst, const int8_t* A, const int8_t* B, float* dst_scale, const float* a_scale,
                   float* b_scale, long channel, long height, long width) {
    OMP_PARALLEL_FOR_GUIDED_
    for (long hw = 0; hw < height * width; hw++) {
        long c = 0;

#ifdef TNN_USE_NEON
        for (; c < channel - 4; c += 8) {
            float32x4_t scale_a_neon0   = vld1q_f32(a_scale + c);
            float32x4_t scale_a_neon1   = vld1q_f32(a_scale + c + 4);
            float32x4_t scale_b_neon0   = vld1q_f32(b_scale + c);
            float32x4_t scale_b_neon1   = vld1q_f32(b_scale + c + 4);
            float32x4_t scale_dst_neon0 = vld1q_f32(dst_scale + c);
            float32x4_t scale_dst_neon1 = vld1q_f32(dst_scale + c + 4);

            long offset        = hw * channel + c;
            int8x8_t aval      = vld1_s8(A + offset);
            int8x8_t bval      = vld1_s8(B + offset);
            int16x8_t a_s16    = vmovl_s8(aval);
            int16x8_t b_s16    = vmovl_s8(bval);
            float32x4_t a0     = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a_s16)));
            float32x4_t b0     = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_s16)));
            float32x4_t a1     = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a_s16)));
            float32x4_t b1     = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_s16)));
            float32x4_t mul0   = vaddq_f32(vmulq_f32(a0, scale_a_neon0), vmulq_f32(b0, scale_b_neon0));
            float32x4_t mul1   = vaddq_f32(vmulq_f32(a1, scale_a_neon1), vmulq_f32(b1, scale_b_neon1));
            int16x4_t mul0_s16 = vqmovn_s32(VCVTAQ_S32_F32(vmulq_f32(mul0, scale_dst_neon0)));
            int16x8_t mul_s16  = VQMOVN_HIGH_S32_T(mul0_s16, VCVTAQ_S32_F32(vmulq_f32(mul1, scale_dst_neon1)));
            vst1_s8(dst + offset, vqmovn_s16(mul_s16));
        }
#endif
        for (; c < channel; c++) {
            long offset = hw * channel + c;
            float aval  = A[offset] * a_scale[c] + B[offset] * b_scale[c];
            dst[offset] = float2int8(aval * dst_scale[c]);
        }
    }
}
void Int8ToFloat(float* dst, const int8_t* src, const float* scale, long batch, long channel, long hw) {
    long c_4 = ROUND_UP(channel, 4);
    for (long n = 0; n < batch; n++) {
        float* dst_c        = dst + n * c_4 * hw;
        const int8_t* src_c = src + n * c_4 * hw;
        OMP_PARALLEL_FOR_GUIDED_
        for (long cnt = 0; cnt < hw; cnt++) {
            long c = 0;
#ifdef TNN_USE_NEON
            for (; c < channel - 4; c += 8) {
                float32x4_t scale_neon0 = vld1q_f32(scale + c);
                float32x4_t scale_neon1 = vld1q_f32(scale + c + 4);
                int8x8_t val            = vld1_s8(src_c + cnt * c_4 + c);
                int16x8_t val_s16       = vmovl_s8(val);
                float32x4_t f32_0       = vcvtq_f32_s32(vmovl_s16(vget_low_s16(val_s16)));
                float32x4_t f32_1       = vcvtq_f32_s32(VMOVL_HIGH_S16_T(val_s16));
                f32_0                   = vmulq_f32(f32_0, scale_neon0);
                f32_1                   = vmulq_f32(f32_1, scale_neon1);
                vst1q_f32(dst_c + cnt * 4 + c * hw, f32_0);
                vst1q_f32(dst_c + cnt * 4 + (c + 4) * hw, f32_1);
            }
#endif
            for (; c < channel; c++) {
                // nchw4 to nhwc4
                long ci                           = c % 4;
                long co                           = c / 4;
                dst_c[co * hw * 4 + cnt * 4 + ci] = static_cast<float>(src_c[cnt * c_4 + c]) * scale[c];
            }
        }
    }
}

void FloatToInt8(int8_t* dst, const float* src, const float* scale, long batch, long channel, long hw) {
#ifdef TNN_USE_NEON
    if (channel <= 4)
        return FloatToInt8C4(dst, src, scale, batch, channel, hw);
#endif
    long c_4 = ROUND_UP(channel, 4);
    for (long n = 0; n < batch; n++) {
        int8_t* dst_c      = dst + n * c_4 * hw;
        const float* src_c = src + n * c_4 * hw;
        OMP_PARALLEL_FOR_GUIDED_
        for (long cnt = 0; cnt < hw; cnt++) {
            // nhwc4 to nchw4
            long idx = 0;
#ifdef TNN_USE_NEON
            idx = channel - channel % 8;

            for (long c = 0; c < idx; c += 8) {
                float32x4_t scale_neon0 = vld1q_f32(scale + c);
                float32x4_t scale_neon1 = vld1q_f32(scale + c + 4);
                float32x4_t val0        = vmulq_f32(vld1q_f32(src_c + c * hw + cnt * 4), scale_neon0);
                float32x4_t val1        = vmulq_f32(vld1q_f32(src_c + (c + 4) * hw + cnt * 4), scale_neon1);
                int16x4_t s16_0         = vqmovn_s32(VCVTAQ_S32_F32(val0));
                int16x8_t s16           = VQMOVN_HIGH_S32_T(s16_0, VCVTAQ_S32_F32(val1));
                vst1_s8(dst_c + cnt * c_4 + c, vqmovn_s16(s16));
            }
#endif
            for (; idx < channel; idx++) {
                long ci                = idx % 4;
                long co                = idx / 4;
                dst_c[cnt * c_4 + idx] = float2int8(src_c[co * hw * 4 + cnt * 4 + ci] * scale[idx]);
            }
        }
    }
}

#ifdef TNN_USE_NEON
/*
assemble kernel used int gemm int8 func
*/
extern "C" {
void GemmInt8Unit4x4(const int8_t* src, const int8_t* weight, int8_t* dst, long src_w_step, long dst_depth, long cdiv8,
                     const float* scale, const int32_t* bias, long relu, const int8_t* add_input,
                     const float* add_scale);
}
#endif

/*
gemm int8 fuse with add func used in linux debug mode
*/
void GemmInt8(int8_t* dst, const int8_t* src, int8_t* work_space, const int8_t* weight, const int32_t* bias,
              const float* scale, long src_depth_d8, long src_w_step, long dst_depth, long relu,
              const int8_t* add_input, const float* add_scale) {
    const long src_depth_d16 = UP_DIV(src_depth_d8, 2);
#if !defined(__aarch64__) && defined(TNN_USE_NEON)
    PackLineV7(src_depth_d8 * 8, reinterpret_cast<const int32_t*>(src), reinterpret_cast<int32_t*>(work_space));
    src = work_space;
#endif
    for (long j = 0; j < dst_depth; j += 4) {
        GemmInt8Unit4x4(src, weight, dst, src_w_step, dst_depth, src_depth_d8, scale + j, bias + j, relu, add_input,
                        add_scale);
        dst += 4;
        weight += 4 * src_depth_d16 * 16;
        if (add_input) {
            add_input += 4;
            add_scale += 4;
        }
    }
}

#ifdef TNN_USE_NEON
inline int16x8x2_t Load16x8x2(const int8_t* src) {
    int8x16_t src_s8 = vld1q_s8(src);
    int16x8x2_t result;
    result.val[0] = vmovl_s8(vget_low_s8(src_s8));
    result.val[1] = vmovl_s8(vget_high_s8(src_s8));
    return result;
}
#endif
/*
gemm int8 func, used in conv int8 common(img2col + gemm)
*/
void GemvInt8(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, const float* scale, long ic_r8,
              long oc_r4) {
#ifdef TNN_USE_NEON
    int8x8_t s8zero = vdup_n_s8(0);
    OMP_PARALLEL_FOR_
    for (long dc = 0; dc < oc_r4; dc += 4) {
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        auto weight_o  = weight + dc * ic_r8;
        long c         = 0;
        for (; c < ic_r8 - 8; c += 16) {
            int16x8x2_t a0 = Load16x8x2(src + c);
            int16x8x2_t b0 = Load16x8x2(weight_o + 0 * ic_r8 + c);
            __builtin_prefetch(weight_o + 0 * ic_r8 + c + 256);
            int16x8x2_t b1 = Load16x8x2(weight_o + 1 * ic_r8 + c);
            __builtin_prefetch(weight_o + 1 * ic_r8 + c + 256);
            int16x8x2_t b2 = Load16x8x2(weight_o + 2 * ic_r8 + c);
            __builtin_prefetch(weight_o + 2 * ic_r8 + c + 256);
            int16x8x2_t b3 = Load16x8x2(weight_o + 3 * ic_r8 + c);
            __builtin_prefetch(weight_o + 3 * ic_r8 + c + 256);
            acc0 = vmlal_s16(acc0, vget_low_s16(a0.val[0]), vget_low_s16(b0.val[0]));
            acc1 = vmlal_s16(acc1, vget_low_s16(a0.val[0]), vget_low_s16(b1.val[0]));
            acc2 = vmlal_s16(acc2, vget_low_s16(a0.val[0]), vget_low_s16(b2.val[0]));
            acc3 = vmlal_s16(acc3, vget_low_s16(a0.val[0]), vget_low_s16(b3.val[0]));

            acc0 = vmlal_s16(acc0, vget_low_s16(a0.val[1]), vget_low_s16(b0.val[1]));
            acc1 = vmlal_s16(acc1, vget_low_s16(a0.val[1]), vget_low_s16(b1.val[1]));
            acc2 = vmlal_s16(acc2, vget_low_s16(a0.val[1]), vget_low_s16(b2.val[1]));
            acc3 = vmlal_s16(acc3, vget_low_s16(a0.val[1]), vget_low_s16(b3.val[1]));

            acc0 = vmlal_s16(acc0, vget_high_s16(a0.val[0]), vget_high_s16(b0.val[0]));
            acc1 = vmlal_s16(acc1, vget_high_s16(a0.val[0]), vget_high_s16(b1.val[0]));
            acc2 = vmlal_s16(acc2, vget_high_s16(a0.val[0]), vget_high_s16(b2.val[0]));
            acc3 = vmlal_s16(acc3, vget_high_s16(a0.val[0]), vget_high_s16(b3.val[0]));

            acc0 = vmlal_s16(acc0, vget_high_s16(a0.val[1]), vget_high_s16(b0.val[1]));
            acc1 = vmlal_s16(acc1, vget_high_s16(a0.val[1]), vget_high_s16(b1.val[1]));
            acc2 = vmlal_s16(acc2, vget_high_s16(a0.val[1]), vget_high_s16(b2.val[1]));
            acc3 = vmlal_s16(acc3, vget_high_s16(a0.val[1]), vget_high_s16(b3.val[1]));
        }

        for (; c < ic_r8; c += 8) {
            int16x8_t a  = vmovl_s8(vld1_s8(src + c));
            int16x8_t b0 = vmovl_s8(vld1_s8(weight_o + 0 * ic_r8 + c));
            int16x8_t b1 = vmovl_s8(vld1_s8(weight_o + 1 * ic_r8 + c));
            int16x8_t b2 = vmovl_s8(vld1_s8(weight_o + 2 * ic_r8 + c));
            int16x8_t b3 = vmovl_s8(vld1_s8(weight_o + 3 * ic_r8 + c));
            acc0         = vmlal_s16(acc0, vget_low_s16(a), vget_low_s16(b0));
            acc1         = vmlal_s16(acc1, vget_low_s16(a), vget_low_s16(b1));
            acc2         = vmlal_s16(acc2, vget_low_s16(a), vget_low_s16(b2));
            acc3         = vmlal_s16(acc3, vget_low_s16(a), vget_low_s16(b3));
            acc0         = vmlal_s16(acc0, vget_high_s16(a), vget_high_s16(b0));
            acc1         = vmlal_s16(acc1, vget_high_s16(a), vget_high_s16(b1));
            acc2         = vmlal_s16(acc2, vget_high_s16(a), vget_high_s16(b2));
            acc3         = vmlal_s16(acc3, vget_high_s16(a), vget_high_s16(b3));
        }
        acc0                  = VPADDQ_S32(acc0, acc1);
        acc2                  = VPADDQ_S32(acc2, acc3);
        int32x4_t acc         = VPADDQ_S32(acc0, acc2);
        int32x4_t bias0       = vld1q_s32(bias + dc);
        float32x4_t scale0    = vld1q_f32(scale + dc);
        *(int32_t*)(dst + dc) = Float4ScaleTos8(vcvtq_f32_s32(vaddq_s32(acc, bias0)), scale0);
    }
#else
    for (long dc = 0; dc < oc_r4; dc++) {
        int32_t acc = bias[dc];
        for (long c = 0; c < ic_r8; c++) {
            acc += src[c] * weight[dc * ic_r8 + c];
        }
        dst[dc] = float2int8(acc * scale[dc]);
    }
#endif
}

/*
convdw int8 kernel, used in corner process
*/
void DepthwiseI8Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, long fw, long fh,
                     long weight_y_step, long dilate_y_step, long dilate_x_step, const float* scale, long dst_depth) {
    long dc = 0;
#ifdef TNN_USE_NEON
    for (; dc < dst_depth - 4; dc += 8) {
        int32x4_t acc0 = vld1q_s32(bias + dc);
        int32x4_t acc1 = vld1q_s32(bias + dc + 4);
        for (long fy = 0; fy < fh; ++fy) {
            const auto src_y    = src + fy * dilate_y_step + dc;
            const auto weight_y = weight + fy * weight_y_step + dc;
            for (long fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilate_x_step;
                const auto weight_x = weight_y + dst_depth * fx;
                int16x8_t a         = vmovl_s8(vld1_s8(src_x));
                int16x8_t b         = vmovl_s8(vld1_s8(weight_x));
                acc0                = vmlal_s16(acc0, vget_low_s16(a), vget_low_s16(b));
                acc1                = vmlal_s16(acc1, vget_high_s16(a), vget_high_s16(b));
            }
        }
        float32x4_t scale0 = vld1q_f32(scale + dc);
        float32x4_t scale1 = vld1q_f32(scale + dc + 4);
        int8x8_t acc_s8    = Float4x2ScaleTos8(vcvtq_f32_s32(acc0), vcvtq_f32_s32(acc1), scale0, scale1);
        vst1_s8(dst + dc, acc_s8);
    }
#endif
    for (; dc < dst_depth; dc += 4) {
        long dst_temp[4] = {0, 0, 0, 0};
        for (long fy = 0; fy < fh; ++fy) {
            const auto src_y    = src + fy * dilate_y_step + dc;
            const auto weight_y = weight + fy * weight_y_step + dc;
            for (long fx = 0; fx < fw; ++fx) {
                const auto weight_x = weight_y + fx * dst_depth;
                const auto src_x    = src_y + fx * dilate_x_step;
                for (long j = 0; j < 4; ++j) {
                    dst_temp[j] += (int32_t)src_x[j] * (int32_t)weight_x[j];
                }
            }
        }
        for (long i = 0; i < 4; ++i) {
            dst[dc + i] = float2int8(static_cast<float>(dst_temp[i] + bias[dc + i]) * scale[dc + i]);
        }
    }
}

/*
general convdw int8 func
*/
void DepthwiseI8General(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
                        long dilate_y_step, long dilate_x_step, long src_w_step, long dst_depth, long fw, long fh,
                        const float* scale_z) {
    long dx, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        long dc = 0;
#ifdef TNN_USE_NEON
        for (; dc < dst_depth - 4; dc += 8) {
            auto dst_x       = dst + dx * dst_depth + dc;
            const auto src_z = src + dx * src_w_step + dc;
            int32x4_t acc0   = vld1q_s32(bias_z + dc);
            int32x4_t acc1   = vld1q_s32(bias_z + dc + 4);

            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilate_y_step;
                const auto weight_y = weight + fy * fw * dst_depth + dc;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilate_x_step;
                    const auto weight_x = weight_y + dst_depth * fx;
                    int16x8_t a         = vmovl_s8(vld1_s8(src_x));
                    int16x8_t b         = vmovl_s8(vld1_s8(weight_x));
                    acc0                = vmlal_s16(acc0, vget_low_s16(a), vget_low_s16(b));
                    acc1                = vmlal_s16(acc1, vget_high_s16(a), vget_high_s16(b));
                }
            }
            float32x4_t scale0 = vld1q_f32(scale_z + dc);
            float32x4_t scale1 = vld1q_f32(scale_z + dc + 4);

            int8x8_t acc_s8 = Float4x2ScaleTos8(vcvtq_f32_s32(acc0), vcvtq_f32_s32(acc1), scale0, scale1);
            vst1_s8(dst_x, acc_s8);
        }
#endif
        for (; dc < dst_depth; dc += 4) {
            auto dst_x          = dst + dx * dst_depth + dc;
            const auto src_z    = src + dx * src_w_step + dc;
            int32_t dstInt32[4] = {0, 0, 0, 0};
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilate_y_step;
                const auto weight_y = weight + fy * fw * dst_depth + dc;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilate_x_step;
                    const auto weight_x = weight_y + dst_depth * fx;
                    for (long j = 0; j < 4; ++j) {
                        dstInt32[j] += (int32_t)src_x[j] * (int32_t)weight_x[j];
                    }
                }
            }

            for (long i = 0; i < 4; ++i) {
                dst_x[i] = float2int8(static_cast<float>(dstInt32[i] + bias_z[i + dc]) * scale_z[i + dc]);
            }
        }
    }
}

#ifdef TNN_USE_NEON
/*
convdw 3x3 int8 func
*/
void DepthwiseI8K3S1Kernel(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
                           long src_y_step, long src_w_step, long dst_depth, long fw, long fh, const float* scale_z,
                           long dx, long dc) {
    auto dst_x       = dst + dx * dst_depth + dc;
    const auto src_z = src + dx * src_w_step + dc;
    int32x4_t acc[4][2];
    int16x8_t a[6], b[3];
    acc[0][0] = vld1q_s32(bias_z + dc);
    acc[0][1] = vld1q_s32(bias_z + dc + 4);
    acc[1][0] = acc[0][0];
    acc[1][1] = acc[0][1];
    acc[2][0] = acc[0][0];
    acc[2][1] = acc[0][1];
    acc[3][0] = acc[0][0];
    acc[3][1] = acc[0][1];

    for (long fy = 0; fy < 3; ++fy) {
        const auto src_y    = src_z + fy * src_y_step;
        const auto weight_y = weight + fy * 3 * dst_depth + dc;
        // unroll loops
        a[0] = vmovl_s8(vld1_s8(src_y + 0 * dst_depth));
        b[0] = vmovl_s8(vld1_s8(weight_y + 0 * dst_depth));
        a[1] = vmovl_s8(vld1_s8(src_y + 1 * dst_depth));
        b[1] = vmovl_s8(vld1_s8(weight_y + 1 * dst_depth));
        a[2] = vmovl_s8(vld1_s8(src_y + 2 * dst_depth));
        b[2] = vmovl_s8(vld1_s8(weight_y + 2 * dst_depth));
        a[3] = vmovl_s8(vld1_s8(src_y + 3 * dst_depth));
        a[4] = vmovl_s8(vld1_s8(src_y + 4 * dst_depth));
        a[5] = vmovl_s8(vld1_s8(src_y + 5 * dst_depth));
        for (long fx = 0; fx < 3; fx++) {
            acc[0][0] = vmlal_s16(acc[0][0], vget_low_s16(a[fx + 0]), vget_low_s16(b[fx]));
            acc[0][1] = vmlal_s16(acc[0][1], vget_high_s16(a[fx + 0]), vget_high_s16(b[fx]));
            acc[1][0] = vmlal_s16(acc[1][0], vget_low_s16(a[fx + 1]), vget_low_s16(b[fx]));
            acc[1][1] = vmlal_s16(acc[1][1], vget_high_s16(a[fx + 1]), vget_high_s16(b[fx]));
            acc[2][0] = vmlal_s16(acc[2][0], vget_low_s16(a[fx + 2]), vget_low_s16(b[fx]));
            acc[2][1] = vmlal_s16(acc[2][1], vget_high_s16(a[fx + 2]), vget_high_s16(b[fx]));
            acc[3][0] = vmlal_s16(acc[3][0], vget_low_s16(a[fx + 3]), vget_low_s16(b[fx]));
            acc[3][1] = vmlal_s16(acc[3][1], vget_high_s16(a[fx + 3]), vget_high_s16(b[fx]));
        }
    }
    float32x4_t scale0 = vld1q_f32(scale_z + dc);
    float32x4_t scale1 = vld1q_f32(scale_z + dc + 4);
    for (long ww = 0; ww < 4; ww++) {
        int8x8_t acc_s8 = Float4x2ScaleTos8(vcvtq_f32_s32(acc[ww][0]), vcvtq_f32_s32(acc[ww][1]), scale0, scale1);
        vst1_s8(dst_x + ww * dst_depth, acc_s8);
    }
}

void DepthwiseI8K3Kernel(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
                         long src_y_step, long src_w_step, long dst_depth, long fw, long fh, const float* scale_z,
                         long dx, long dc) {
    auto dst_x       = dst + dx * dst_depth + dc;
    const auto src_z = src + dx * src_w_step + dc;
    int32x4_t acc0   = vld1q_s32(bias_z + dc);
    int32x4_t acc1   = vld1q_s32(bias_z + dc + 4);

    for (long fy = 0; fy < 3; ++fy) {
        const auto src_y    = src_z + fy * src_y_step;
        const auto weight_y = weight + fy * 3 * dst_depth + dc;
        int16x8_t a[3], b[3];
        a[0] = vmovl_s8(vld1_s8(src_y + 0 * dst_depth));
        b[0] = vmovl_s8(vld1_s8(weight_y + 0 * dst_depth));
        a[1] = vmovl_s8(vld1_s8(src_y + 1 * dst_depth));
        b[1] = vmovl_s8(vld1_s8(weight_y + 1 * dst_depth));
        a[2] = vmovl_s8(vld1_s8(src_y + 2 * dst_depth));
        b[2] = vmovl_s8(vld1_s8(weight_y + 2 * dst_depth));
        acc0 = vmlal_s16(acc0, vget_low_s16(a[0]), vget_low_s16(b[0]));
        acc1 = vmlal_s16(acc1, vget_high_s16(a[0]), vget_high_s16(b[0]));
        acc0 = vmlal_s16(acc0, vget_low_s16(a[1]), vget_low_s16(b[1]));
        acc1 = vmlal_s16(acc1, vget_high_s16(a[1]), vget_high_s16(b[1]));
        acc0 = vmlal_s16(acc0, vget_low_s16(a[2]), vget_low_s16(b[2]));
        acc1 = vmlal_s16(acc1, vget_high_s16(a[2]), vget_high_s16(b[2]));
    }
    float32x4_t scale0 = vld1q_f32(scale_z + dc);
    float32x4_t scale1 = vld1q_f32(scale_z + dc + 4);

    int8x8_t acc_s8 = Float4x2ScaleTos8(vcvtq_f32_s32(acc0), vcvtq_f32_s32(acc1), scale0, scale1);
    vst1_s8(dst_x, acc_s8);
}

void DepthwiseI8K3(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
                   long dilate_y_step, long dialte_x_step, long src_w_step, long dst_depth, long fw, long fh,
                   const float* scale_z) {
    long dx = 0;
    // todo:3x8 for arm v7 16regs
    // stride == 1, fully use arm registers
    if (src_w_step == dst_depth) {
        for (dx = 0; dx < width - 3; dx += 4) {
            long dc = 0;
            for (; dc < dst_depth - 7; dc += 8) {
                DepthwiseI8K3S1Kernel(dst, src, weight, bias_z, width, dilate_y_step, src_w_step, dst_depth, fw, fh,
                                      scale_z, dx, dc);
            }

            if (dc < dst_depth) {
                dc = dst_depth - 8;
                DepthwiseI8K3S1Kernel(dst, src, weight, bias_z, width, dilate_y_step, src_w_step, dst_depth, fw, fh,
                                      scale_z, dx, dc);
            }
        }
    }

    // general k3 process, calc left dx
    for (; dx < width; dx++) {
        long dc = 0;
        for (; dc < dst_depth - 7; dc += 8) {
            DepthwiseI8K3Kernel(dst, src, weight, bias_z, width, dilate_y_step, src_w_step, dst_depth, fw, fh, scale_z,
                                dx, dc);
        }

        if (dc < dst_depth) {
            dc = dst_depth - 8;
            DepthwiseI8K3Kernel(dst, src, weight, bias_z, width, dilate_y_step, src_w_step, dst_depth, fw, fh, scale_z,
                                dx, dc);
        }
    }
}
#endif

void ReluInt8(int8_t* dst, const int8_t* src, long len) {
    long idx = 0;
#ifdef TNN_USE_NEON
    int8x8_t zero = vdup_n_s8(0);
    idx           = len - len % 8;
    OMP_PARALLEL_FOR_GUIDED_
    for (long i = 0; i < idx; i += 8) {
        int8x8_t val = vld1_s8(src + i);
        vst1_s8(dst + i, vmax_s8(val, zero));
    }
#endif
    for (; idx < len; idx++) {
        dst[idx] = MAX(0, src[idx]);
    }
}

}  // namespace TNN_NS
