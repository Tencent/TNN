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

#if __arm__
#include "indirect_conv_int8_4x8.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/omp_utils.h"

#define ASMCONVINT8UNIT4X8
#ifdef ASMCONVINT8UNIT4X8
extern "C" {
void ASMConvInt8Unit4x8(int32_t mr, int32_t nr, int32_t kc, int32_t ks, const int32_t* a, const void* w, int8_t* c,
                        int32_t c_stride, const float* scales, int32_t relu, const int8_t* add_input,
                        const float* add_scale, const int8_t* zero, const int8_t* real_input);
}
#endif
namespace TNN_NS {
#define COMPUTE_UNIT_LOW4(i)                                                                                           \
    {                                                                                                                  \
        const int16x8_t vb01234567 = vmovl_s8(vld1_s8((const int8_t*)weight));                                         \
        weight                     = (void*)((uintptr_t)weight + 8);                                                   \
        vacc0x0123                 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb01234567), vget_low_s16(va0), i);       \
        vacc0x4567                 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vb01234567), vget_low_s16(va0), i);      \
        vacc1x0123                 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb01234567), vget_low_s16(va1), i);       \
        vacc1x4567                 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vb01234567), vget_low_s16(va1), i);      \
        vacc2x0123                 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb01234567), vget_low_s16(va2), i);       \
        vacc2x4567                 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vb01234567), vget_low_s16(va2), i);      \
        vacc3x0123                 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb01234567), vget_low_s16(va3), i);       \
        vacc3x4567                 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vb01234567), vget_low_s16(va3), i);      \
    }

#define COMPUTE_UNIT_HIGH4(i)                                                                                          \
    {                                                                                                                  \
        const int16x8_t vb01234567 = vmovl_s8(vld1_s8((const int8_t*)weight));                                         \
        weight                     = (void*)((uintptr_t)weight + 8);                                                   \
        vacc0x0123                 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vb01234567), vget_high_s16(va0), i);      \
        vacc0x4567                 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vb01234567), vget_high_s16(va0), i);     \
        vacc1x0123                 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vb01234567), vget_high_s16(va1), i);      \
        vacc1x4567                 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vb01234567), vget_high_s16(va1), i);     \
        vacc2x0123                 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vb01234567), vget_high_s16(va2), i);      \
        vacc2x4567                 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vb01234567), vget_high_s16(va2), i);     \
        vacc3x0123                 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vb01234567), vget_high_s16(va3), i);      \
        vacc3x4567                 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vb01234567), vget_high_s16(va3), i);     \
    }

void IndirectConvInt8Unit4x8(int32_t mr, int32_t nr, int32_t input_channel, int32_t kernel_size,
                             const int32_t* indirect, const void* weight, int8_t* output, int32_t channel_stride,
                             const float* scales, int32_t relu, const int8_t* add_input, const float* add_scale,
                             const int8_t* zero, const int8_t* real_input) {
#if !defined(ASMCONVINT8UNIT4X8)
    union {
        const void* as_void_ptr;
        int8_t* as_int8_ptr;
        int32_t* as_int32_ptr;
    } packed             = {weight};
    int32x4_t vacc0x0123 = vld1q_s32((const int32_t*)weight);
    weight               = (void*)((uintptr_t)weight + 16);
    int32x4_t vacc0x4567 = vld1q_s32((const int32_t*)weight);
    weight               = (void*)((uintptr_t)weight + 16);
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;
    do {
        const int8_t* a0 = *indirect == -1 ? zero : (real_input + *indirect);
        indirect++;
        const int8_t* a1 = *indirect == -1 ? zero : (real_input + *indirect);
        indirect++;
        const int8_t* a2 = *indirect == -1 ? zero : (real_input + *indirect);
        indirect++;
        const int8_t* a3 = *indirect == -1 ? zero : (real_input + *indirect);
        indirect++;
        long k = input_channel;
        for (; k >= 8; k -= 8) {
            const int16x8_t va0 = vmovl_s8(vld1_s8(a0));
            a0 += 8;
            const int16x8_t va1 = vmovl_s8(vld1_s8(a1));
            a1 += 8;
            const int16x8_t va2 = vmovl_s8(vld1_s8(a2));
            a2 += 8;
            const int16x8_t va3 = vmovl_s8(vld1_s8(a3));
            a3 += 8;
            COMPUTE_UNIT_LOW4(0);
            COMPUTE_UNIT_LOW4(1);
            COMPUTE_UNIT_LOW4(2);
            COMPUTE_UNIT_LOW4(3);
            COMPUTE_UNIT_HIGH4(0);
            COMPUTE_UNIT_HIGH4(1);
            COMPUTE_UNIT_HIGH4(2);
            COMPUTE_UNIT_HIGH4(3);
        }
        if (k != 0) {
            const int32_t a_predecrement = 8 - k;
            const int64x1_t va_shift     = vmov_n_s64(-8 * a_predecrement);
            const int16x8_t va0 =
                vmovl_s8(vreinterpret_s8_s64(vshl_s64(vreinterpret_s64_s8(vld1_s8(a0 - a_predecrement)), va_shift)));
            const int16x8_t va1 =
                vmovl_s8(vreinterpret_s8_s64(vshl_s64(vreinterpret_s64_s8(vld1_s8(a1 - a_predecrement)), va_shift)));
            const int16x8_t va2 =
                vmovl_s8(vreinterpret_s8_s64(vshl_s64(vreinterpret_s64_s8(vld1_s8(a2 - a_predecrement)), va_shift)));
            const int16x8_t va3 =
                vmovl_s8(vreinterpret_s8_s64(vshl_s64(vreinterpret_s64_s8(vld1_s8(a3 - a_predecrement)), va_shift)));
            COMPUTE_UNIT_LOW4(0);

            if (k >= 2) {
                COMPUTE_UNIT_LOW4(1);
                if (k > 2) {
                    COMPUTE_UNIT_LOW4(2);
                    if (k >= 4) {
                        COMPUTE_UNIT_LOW4(3);
                        if (k > 4) {
                            COMPUTE_UNIT_HIGH4(0);
                            if (k >= 6) {
                                COMPUTE_UNIT_HIGH4(1);
                                if (k > 6) {
                                    COMPUTE_UNIT_HIGH4(2);
                                }
                            }
                        }
                    }
                }
            }
        }

    } while (--kernel_size != 0);
    const float32x4_t vscale0123 = vld1q_f32(scales);
    const float32x4_t vscale4567 = nr > 4 ? vld1q_f32(scales + 4) : vdupq_n_f32(0.f);
    float32x4_t vfacc0x0123      = vmulq_f32(vcvtq_f32_s32(vacc0x0123), vscale0123);
    float32x4_t vfacc1x0123      = vmulq_f32(vcvtq_f32_s32(vacc1x0123), vscale0123);
    float32x4_t vfacc2x0123      = vmulq_f32(vcvtq_f32_s32(vacc2x0123), vscale0123);
    float32x4_t vfacc3x0123      = vmulq_f32(vcvtq_f32_s32(vacc3x0123), vscale0123);
    float32x4_t vfacc0x4567      = vmulq_f32(vcvtq_f32_s32(vacc0x4567), vscale4567);
    float32x4_t vfacc1x4567      = vmulq_f32(vcvtq_f32_s32(vacc1x4567), vscale4567);
    float32x4_t vfacc2x4567      = vmulq_f32(vcvtq_f32_s32(vacc2x4567), vscale4567);
    float32x4_t vfacc3x4567      = vmulq_f32(vcvtq_f32_s32(vacc3x4567), vscale4567);

    if (relu < 0) {
        float32x4_t vzero = vdupq_n_f32(0);
        vfacc0x0123       = vmaxq_f32(vfacc0x0123, vzero);
        vfacc1x0123       = vmaxq_f32(vfacc1x0123, vzero);
        vfacc2x0123       = vmaxq_f32(vfacc2x0123, vzero);
        vfacc3x0123       = vmaxq_f32(vfacc3x0123, vzero);
        vfacc0x4567       = vmaxq_f32(vfacc0x4567, vzero);
        vfacc1x4567       = vmaxq_f32(vfacc1x4567, vzero);
        vfacc2x4567       = vmaxq_f32(vfacc2x4567, vzero);
        vfacc3x4567       = vmaxq_f32(vfacc3x4567, vzero);
    }
    if (add_input) {
        const float32x4_t vaddscale0123 = vld1q_f32(add_scale);
        const float32x4_t vaddscale4567 = nr > 4 ? vld1q_f32(add_scale + 4) : vdupq_n_f32(0.f);
        const int8_t* add_input1        = (const int8_t*)((uintptr_t)add_input + channel_stride);
        if (mr < 2) {
            add_input1 = add_input;
        }
        const int8_t* add_input2 = (const int8_t*)((uintptr_t)add_input1 + channel_stride);
        if (mr <= 2) {
            add_input2 = add_input1;
        }
        const int8_t* add_input3 = (const int8_t*)((uintptr_t)add_input2 + channel_stride);
        if (mr != 4) {
            add_input3 = add_input2;
        }
        int16x8_t vaddinput0x01234567 = vmovl_s8(vld1_s8(add_input));
        int16x8_t vaddinput1x01234567 = vmovl_s8(vld1_s8(add_input1));
        int16x8_t vaddinput2x01234567 = vmovl_s8(vld1_s8(add_input2));
        int16x8_t vaddinput3x01234567 = vmovl_s8(vld1_s8(add_input3));
        vfacc0x0123 =
            vmlaq_f32(vfacc0x0123, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vaddinput0x01234567))), vaddscale0123);
        vfacc1x0123 =
            vmlaq_f32(vfacc1x0123, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vaddinput1x01234567))), vaddscale0123);
        vfacc2x0123 =
            vmlaq_f32(vfacc2x0123, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vaddinput2x01234567))), vaddscale0123);
        vfacc3x0123 =
            vmlaq_f32(vfacc3x0123, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vaddinput3x01234567))), vaddscale0123);
        vfacc0x4567 =
            vmlaq_f32(vfacc0x4567, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vaddinput0x01234567))), vaddscale4567);
        vfacc1x4567 =
            vmlaq_f32(vfacc1x4567, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vaddinput1x01234567))), vaddscale4567);
        vfacc2x4567 =
            vmlaq_f32(vfacc2x4567, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vaddinput2x01234567))), vaddscale4567);
        vfacc3x4567 =
            vmlaq_f32(vfacc3x4567, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vaddinput3x01234567))), vaddscale4567);
    }
    vacc0x0123 = VCVTAQ_S32_F32(vfacc0x0123);
    vacc1x0123 = VCVTAQ_S32_F32(vfacc1x0123);
    vacc2x0123 = VCVTAQ_S32_F32(vfacc2x0123);
    vacc3x0123 = VCVTAQ_S32_F32(vfacc3x0123);
    vacc0x4567 = VCVTAQ_S32_F32(vfacc0x4567);
    vacc1x4567 = VCVTAQ_S32_F32(vfacc1x4567);
    vacc2x4567 = VCVTAQ_S32_F32(vfacc2x4567);
    vacc3x4567 = VCVTAQ_S32_F32(vfacc3x4567);

    int8x8_t vacc0x01234567 = vqmovn_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)));
    int8x8_t vacc1x01234567 = vqmovn_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)));
    int8x8_t vacc2x01234567 = vqmovn_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)));
    int8x8_t vacc3x01234567 = vqmovn_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)));

    if (relu > 0) {
        int8x8_t vzero = vdup_n_s8(0);
        vacc0x01234567 = vmax_s8(vacc0x01234567, vzero);
        vacc1x01234567 = vmax_s8(vacc1x01234567, vzero);
        vacc2x01234567 = vmax_s8(vacc2x01234567, vzero);
        vacc3x01234567 = vmax_s8(vacc3x01234567, vzero);
    }

    int8_t* c0 = output;
    int8_t* c1 = (int8_t*)((uintptr_t)c0 + channel_stride);
    if (mr < 2) {
        c1 = c0;
    }
    int8_t* c2 = (int8_t*)((uintptr_t)c1 + channel_stride);
    if (mr <= 2) {
        c2 = c1;
    }
    int8_t* c3 = (int8_t*)((uintptr_t)c2 + channel_stride);
    if (mr != 4) {
        c3 = c2;
    }
    if (nr == 8) {
        vst1_s8(c0, vacc0x01234567);
        vst1_s8(c1, vacc1x01234567);
        vst1_s8(c2, vacc2x01234567);
        vst1_s8(c3, vacc3x01234567);
    } else {
        if (nr >= 4) {
            vst1_lane_s32((int32_t*)c0, vreinterpret_s32_s8(vacc0x01234567), 0);
            c0 += 4;
            vst1_lane_s32((int32_t*)c1, vreinterpret_s32_s8(vacc1x01234567), 0);
            c1 += 4;
            vst1_lane_s32((int32_t*)c2, vreinterpret_s32_s8(vacc2x01234567), 0);
            c2 += 4;
            vst1_lane_s32((int32_t*)c3, vreinterpret_s32_s8(vacc3x01234567), 0);
            c3 += 4;
            vacc0x01234567 = vext_s8(vacc0x01234567, vacc0x01234567, 4);
            vacc1x01234567 = vext_s8(vacc1x01234567, vacc1x01234567, 4);
            vacc2x01234567 = vext_s8(vacc2x01234567, vacc2x01234567, 4);
            vacc3x01234567 = vext_s8(vacc3x01234567, vacc3x01234567, 4);
            nr -= 4;
        }
        if (nr >= 2) {
            vst1_lane_s16((int16_t*)c0, vreinterpret_s16_s8(vacc0x01234567), 0);
            c0 += 2;
            vst1_lane_s16((int16_t*)c1, vreinterpret_s16_s8(vacc1x01234567), 0);
            c1 += 2;
            vst1_lane_s16((int16_t*)c2, vreinterpret_s16_s8(vacc2x01234567), 0);
            c2 += 2;
            vst1_lane_s16((int16_t*)c3, vreinterpret_s16_s8(vacc3x01234567), 0);
            c3 += 2;
            vacc0x01234567 = vext_s8(vacc0x01234567, vacc0x01234567, 2);
            vacc1x01234567 = vext_s8(vacc1x01234567, vacc1x01234567, 2);
            vacc2x01234567 = vext_s8(vacc2x01234567, vacc2x01234567, 2);
            vacc3x01234567 = vext_s8(vacc3x01234567, vacc3x01234567, 2);
            nr -= 2;
        }
        if (nr != 0) {
            vst1_lane_s8(c0, vacc0x01234567, 0);
            vst1_lane_s8(c1, vacc1x01234567, 0);
            vst1_lane_s8(c2, vacc2x01234567, 0);
            vst1_lane_s8(c3, vacc3x01234567, 0);
        }
    }
#else
    ASMConvInt8Unit4x8(mr, nr, input_channel, kernel_size, indirect, weight, output, channel_stride, scales, relu,
                       add_input, add_scale, zero, real_input);
#endif
}
}  // namespace TNN_NS
#endif