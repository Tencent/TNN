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

#include "tnn/device/arm/arm_blob_converter.h"

#include "tnn/core/blob_int8.h"
#include "tnn/core/macro.h"
#include "tnn/device/arm/acc/Float4.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_util.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

ArmBlobConverterAcc::ArmBlobConverterAcc(Blob *blob) : BlobConverterAcc(blob) {}
ArmBlobConverterAcc::~ArmBlobConverterAcc() {}

/*
convert data type from  Tin to Tout, data format from nc4hw4 2 nchw
*/
template <typename Tin, typename Tout>
void FloatBlobToNCHW(const Tin *src, Tout *dst, int channel, int hw) {
    if (channel % 4 == 0 && hw == 1 && sizeof(Tin) == sizeof(Tout)) {
        memcpy(dst, src, channel * sizeof(Tin));
        return;
    }
    UnpackC4(dst, src, hw, channel);
    return;
}

template void FloatBlobToNCHW(const float *src, bfp16_t *dst, int channel, int hw);
template void FloatBlobToNCHW(const float *src, float *dst, int channel, int hw);
template void FloatBlobToNCHW(const bfp16_t *src, float *dst, int channel, int hw);
template void FloatBlobToNCHW(const bfp16_t *src, bfp16_t *dst, int channel, int hw);

/*
convert data type from int8 to float, data format from nhwc 2 nchw
*/
static void Int8BlobToNCHW(const int8_t *src, float *dst, int channel, int hw, float *scale, float *bias) {
    UnpackAndDequant(dst, src, hw, channel, scale, bias);
    return;
}

/*
convert data type from uint8 to float, data format from nhwc 2 nchw
*/
template <bool reverse_channel>
static void BGRAToBlobImpl(const uint8_t *src, float *dst, float *scale, float *bias, int hw, int channel) {
    int i = 0;
#ifdef TNN_USE_NEON
    float32x4_t bias_neon_b = vdupq_n_f32(bias[0]);
    float32x4_t bias_neon_g = vdupq_n_f32(bias[1]);
    float32x4_t bias_neon_r = vdupq_n_f32(bias[2]);
    float32x4_t bias_neon_a = vdupq_n_f32(bias[3]);
    float32x4x4_t vf32;
    for (; i < hw - 7; i += 8) {
        uint8x8x4_t v_u8 = vld4_u8(src + i * 4);
        int16x8_t b_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[0]));
        int16x8_t g_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[1]));
        int16x8_t r_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[2]));
        int16x8_t a_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[3]));

        vf32.val[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(reverse_channel ? r_s16 : b_s16)));
        vf32.val[1] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(g_s16)));
        vf32.val[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(reverse_channel ? b_s16 : r_s16)));
        vf32.val[3] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a_s16)));

        vf32.val[0] = vaddq_f32(bias_neon_b, vmulq_n_f32(vf32.val[0], scale[0]));
        vf32.val[1] = vaddq_f32(bias_neon_g, vmulq_n_f32(vf32.val[1], scale[1]));
        vf32.val[2] = vaddq_f32(bias_neon_r, vmulq_n_f32(vf32.val[2], scale[2]));
        vf32.val[3] = vaddq_f32(bias_neon_a, vmulq_n_f32(vf32.val[3], scale[3]));

        if (channel == 3) {
            vf32.val[3] = vdupq_n_f32(0.0f);
        }

        vst4q_f32(dst + i * 4, vf32);

        vf32.val[0] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(reverse_channel ? r_s16 : b_s16)));
        vf32.val[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(g_s16)));
        vf32.val[2] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(reverse_channel ? b_s16 : r_s16)));
        vf32.val[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a_s16)));

        vf32.val[0] = vaddq_f32(bias_neon_b, vmulq_n_f32(vf32.val[0], scale[0]));
        vf32.val[1] = vaddq_f32(bias_neon_g, vmulq_n_f32(vf32.val[1], scale[1]));
        vf32.val[2] = vaddq_f32(bias_neon_r, vmulq_n_f32(vf32.val[2], scale[2]));
        vf32.val[3] = vaddq_f32(bias_neon_a, vmulq_n_f32(vf32.val[3], scale[3]));

        if (channel == 3) {
            vf32.val[3] = vdupq_n_f32(0.0f);
        }

        vst4q_f32(dst + i * 4 + 16, vf32);
    }
#endif
    for (; i < hw; ++i) {
        dst[4 * i + 0] = scale[0] * src[4 * i + (reverse_channel ? 2 : 0)] + bias[0];
        dst[4 * i + 1] = scale[1] * src[4 * i + 1] + bias[1];
        dst[4 * i + 2] = scale[2] * src[4 * i + (reverse_channel ? 0 : 2)] + bias[2];
        dst[4 * i + 3] = scale[3] * src[4 * i + 3] + bias[3];
        if (channel == 3) {
            dst[4 * i + 3] = 0.0f;
        }
    }
}

/*
convert data type from uint8 to float, data format from nhw4 2 nc4hw4
*/
template <bool reverse_channel>
static void BGRAToBlobImpl(const uint8_t *src, int8_t *dst, float *scale, float *bias, int hw, int channel) {
    int i = 0;
#ifdef TNN_USE_NEON
    float32x4_t bias_neon_b = vdupq_n_f32(bias[0]);
    float32x4_t bias_neon_g = vdupq_n_f32(bias[1]);
    float32x4_t bias_neon_r = vdupq_n_f32(bias[2]);
    float32x4_t bias_neon_a = vdupq_n_f32(bias[3]);
    int8x8x4_t vi8x4;
    for (; i < hw - 7; i += 8) {
        uint8x8x4_t v_u8 = vld4_u8(src + i * 4);
        int16x8_t b_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[0]));
        int16x8_t g_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[1]));
        int16x8_t r_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[2]));
        int16x8_t a_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[3]));

        float32x4_t f32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(reverse_channel ? r_s16 : b_s16)));
        float32x4_t f32_1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(g_s16)));
        float32x4_t f32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(reverse_channel ? b_s16 : r_s16)));
        float32x4_t f32_3 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a_s16)));
        float32x4_t f32_4 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(reverse_channel ? r_s16 : b_s16)));
        float32x4_t f32_5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(g_s16)));
        float32x4_t f32_6 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(reverse_channel ? b_s16 : r_s16)));
        float32x4_t f32_7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a_s16)));

        f32_0 = vaddq_f32(bias_neon_b, vmulq_n_f32(f32_0, scale[0]));
        f32_1 = vaddq_f32(bias_neon_g, vmulq_n_f32(f32_1, scale[1]));
        f32_2 = vaddq_f32(bias_neon_r, vmulq_n_f32(f32_2, scale[2]));
        f32_3 = vaddq_f32(bias_neon_a, vmulq_n_f32(f32_3, scale[3]));
        f32_4 = vaddq_f32(bias_neon_b, vmulq_n_f32(f32_4, scale[0]));
        f32_5 = vaddq_f32(bias_neon_g, vmulq_n_f32(f32_5, scale[1]));
        f32_6 = vaddq_f32(bias_neon_r, vmulq_n_f32(f32_6, scale[2]));
        f32_7 = vaddq_f32(bias_neon_a, vmulq_n_f32(f32_7, scale[3]));

        int16x4_t s16_l0 = vqmovn_s32(VCVTAQ_S32_F32(f32_0));
        int16x8_t s16_0  = VQMOVN_HIGH_S32_T(s16_l0, VCVTAQ_S32_F32(f32_4));
        int16x4_t s16_l1 = vqmovn_s32(VCVTAQ_S32_F32(f32_1));
        int16x8_t s16_1  = VQMOVN_HIGH_S32_T(s16_l1, VCVTAQ_S32_F32(f32_5));
        int16x4_t s16_l2 = vqmovn_s32(VCVTAQ_S32_F32(f32_2));
        int16x8_t s16_2  = VQMOVN_HIGH_S32_T(s16_l2, VCVTAQ_S32_F32(f32_6));
        int16x4_t s16_l3 = vqmovn_s32(VCVTAQ_S32_F32(f32_3));
        int16x8_t s16_3  = VQMOVN_HIGH_S32_T(s16_l3, VCVTAQ_S32_F32(f32_7));

        vi8x4.val[0] = vqmovn_s16(s16_0);
        vi8x4.val[1] = vqmovn_s16(s16_1);
        vi8x4.val[2] = vqmovn_s16(s16_2);
        vi8x4.val[3] = vqmovn_s16(s16_3);

        if (channel == 3) {
            vi8x4.val[3] = vdup_n_s8(0);
        }

        vst4_s8(dst + i * 4, vi8x4);
    }
#endif
    for (; i < hw; ++i) {
        dst[4 * i + 0] = float2int8(scale[0] * src[4 * i + (reverse_channel ? 2 : 0)] + bias[0]);
        dst[4 * i + 1] = float2int8(scale[1] * src[4 * i + 1] + bias[1]);
        dst[4 * i + 2] = float2int8(scale[2] * src[4 * i + (reverse_channel ? 0 : 2)] + bias[2]);
        dst[4 * i + 3] = float2int8(scale[3] * src[4 * i + 3] + bias[3]);
        if (channel == 3) {
            dst[4 * i + 3] = 0;
        }
    }
}

/*
if channel == 3, the fourth channel is ignored
*/
template<typename T>
static void BGRAToBlob(const uint8_t *src, T *dst, float *scale, float *bias, int hw,
                       bool reverse_channel, int channel) {
    if (reverse_channel) {
        BGRAToBlobImpl<true>(src, dst, scale, bias, hw, channel);
    } else {
        BGRAToBlobImpl<false>(src, dst, scale, bias, hw, channel);
    }
}

/*
convert data type from uint8 to float, data format from nhw1 2 nc4hw4
*/
static void GrayToBlob(const uint8_t *src, float *dst, float scale, float bias, int hw) {
    int i = 0;
    memset(dst, 0, hw * 4 * sizeof(float));
#ifdef TNN_USE_NEON
    float32x4_t scale_neon = vdupq_n_f32(scale);
    float32x4_t bias_neon  = vdupq_n_f32(bias);
    for (; i < hw - 7; i += 8) {
        uint8x8_t v_u8     = vld1_u8(src + i);
        int16x8_t v_s16    = vreinterpretq_s16_u16(vmovl_u8(v_u8));
        float32x4_t vf32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_s16)));
        float32x4_t vf32_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_s16)));
        float32x4_t rf32_0 = vaddq_f32(bias_neon, vmulq_f32(scale_neon, vf32_0));
        float32x4_t rf32_1 = vaddq_f32(bias_neon, vmulq_f32(scale_neon, vf32_1));

        dst[(i + 0) * 4] = vgetq_lane_f32(rf32_0, 0);
        dst[(i + 1) * 4] = vgetq_lane_f32(rf32_0, 1);
        dst[(i + 2) * 4] = vgetq_lane_f32(rf32_0, 2);
        dst[(i + 3) * 4] = vgetq_lane_f32(rf32_0, 3);
        dst[(i + 4) * 4] = vgetq_lane_f32(rf32_1, 0);
        dst[(i + 5) * 4] = vgetq_lane_f32(rf32_1, 1);
        dst[(i + 6) * 4] = vgetq_lane_f32(rf32_1, 2);
        dst[(i + 7) * 4] = vgetq_lane_f32(rf32_1, 3);
    }
#endif
    for (; i < hw; ++i) {
        dst[4 * i] = scale * src[i] + bias;
    }
}

/*
convert data type from uint8 to int8, data format from nhw1 2 nhwc
*/
static void GrayToBlob(const uint8_t *src, int8_t *dst, float scale, float bias, int hw) {
    int i = 0;
    memset(dst, 0, hw * 4 * sizeof(int8_t));
#ifdef TNN_USE_NEON
    float32x4_t bias_neon = vdupq_n_f32(bias);
    int8_t dst_tmp[8];
    for (; i < hw - 7; i += 8) {
        uint8x8_t v_u8     = vld1_u8(src + i);
        int16x8_t v_s16    = vreinterpretq_s16_u16(vmovl_u8(v_u8));
        float32x4_t vf32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_s16)));
        float32x4_t vf32_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_s16)));
        float32x4_t rf32_0 = vaddq_f32(bias_neon, vmulq_n_f32(vf32_0, scale));
        float32x4_t rf32_1 = vaddq_f32(bias_neon, vmulq_n_f32(vf32_1, scale));

        int16x4_t s16_l = vqmovn_s32(VCVTAQ_S32_F32(rf32_0));
        int16x8_t s16   = VQMOVN_HIGH_S32_T(s16_l, VCVTAQ_S32_F32(rf32_1));
        vst1_s8(dst_tmp, vqmovn_s16(s16));
        dst[(i + 0) * 4] = dst_tmp[0];
        dst[(i + 1) * 4] = dst_tmp[1];
        dst[(i + 2) * 4] = dst_tmp[2];
        dst[(i + 3) * 4] = dst_tmp[3];
        dst[(i + 4) * 4] = dst_tmp[4];
        dst[(i + 5) * 4] = dst_tmp[5];
        dst[(i + 6) * 4] = dst_tmp[6];
        dst[(i + 7) * 4] = dst_tmp[7];
    }
#endif
    for (; i < hw; ++i) {
        dst[4 * i] = float2int8(scale * src[i] + bias);
    }
}

/*
convert data type from uint8 to float, data format from nhw3 2 nc4hw4
*/
template <bool reverse_channel>
static void BGRToBlobImpl(const uint8_t *src, float *dst, float *scale, float *bias, int hw) {
    int i = 0;
#ifdef TNN_USE_NEON
    float32x4_t bias_neon_b = vdupq_n_f32(bias[0]);
    float32x4_t bias_neon_g = vdupq_n_f32(bias[1]);
    float32x4_t bias_neon_r = vdupq_n_f32(bias[2]);
    float32x4x4_t vf32;
    vf32.val[3] = vdupq_n_f32(0);
    for (; i < hw - 7; i += 8) {
        uint8x8x3_t v_u8 = vld3_u8(src + i * 3);
        int16x8_t b_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[0]));
        int16x8_t g_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[1]));
        int16x8_t r_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[2]));

        vf32.val[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(reverse_channel ? r_s16 : b_s16)));
        vf32.val[1] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(g_s16)));
        vf32.val[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(reverse_channel ? b_s16 : r_s16)));

        vf32.val[0] = vaddq_f32(bias_neon_b, vmulq_n_f32(vf32.val[0], scale[0]));
        vf32.val[1] = vaddq_f32(bias_neon_g, vmulq_n_f32(vf32.val[1], scale[1]));
        vf32.val[2] = vaddq_f32(bias_neon_r, vmulq_n_f32(vf32.val[2], scale[2]));

        vst4q_f32(dst + i * 4, vf32);

        vf32.val[0] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(reverse_channel ? r_s16 : b_s16)));
        vf32.val[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(g_s16)));
        vf32.val[2] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(reverse_channel ? b_s16 : r_s16)));

        vf32.val[0] = vaddq_f32(bias_neon_b, vmulq_n_f32(vf32.val[0], scale[0]));
        vf32.val[1] = vaddq_f32(bias_neon_g, vmulq_n_f32(vf32.val[1], scale[1]));
        vf32.val[2] = vaddq_f32(bias_neon_r, vmulq_n_f32(vf32.val[2], scale[2]));

        vst4q_f32(dst + i * 4 + 16, vf32);
    }
#endif
    for (; i < hw; ++i) {
        dst[4 * i + 0] = scale[0] * src[3 * i + (reverse_channel ? 2 : 0)] + bias[0];
        dst[4 * i + 1] = scale[1] * src[3 * i + 1] + bias[1];
        dst[4 * i + 2] = scale[2] * src[3 * i + (reverse_channel ? 0 : 2)] + bias[2];
        dst[4 * i + 3] = 0;
    }
}

/*
convert data type from uint8 to float, data format from nhw3 2 nc4hw4
*/
template <bool reverse_channel>
static void BGRToBlobImpl(const uint8_t *src, int8_t *dst, float *scale, float *bias, int hw) {
    int i = 0;
#ifdef TNN_USE_NEON
    float32x4_t bias_neon_b = vdupq_n_f32(bias[0]);
    float32x4_t bias_neon_g = vdupq_n_f32(bias[1]);
    float32x4_t bias_neon_r = vdupq_n_f32(bias[2]);
    int8x8x4_t vi8x4;
    vi8x4.val[3] = vdup_n_s8(0);
    for (; i < hw - 7; i += 8) {
        uint8x8x3_t v_u8 = vld3_u8(src + i * 3);
        int16x8_t b_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[0]));
        int16x8_t g_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[1]));
        int16x8_t r_s16  = vreinterpretq_s16_u16(vmovl_u8(v_u8.val[2]));

        float32x4_t f32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(reverse_channel ? r_s16 : b_s16)));
        float32x4_t f32_1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(g_s16)));
        float32x4_t f32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(reverse_channel ? b_s16 : r_s16)));
        float32x4_t f32_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(reverse_channel ? r_s16 : b_s16)));
        float32x4_t f32_4 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(g_s16)));
        float32x4_t f32_5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(reverse_channel ? b_s16 : r_s16)));

        f32_0 = vaddq_f32(bias_neon_b, vmulq_n_f32(f32_0, scale[0]));
        f32_1 = vaddq_f32(bias_neon_g, vmulq_n_f32(f32_1, scale[1]));
        f32_2 = vaddq_f32(bias_neon_r, vmulq_n_f32(f32_2, scale[2]));
        f32_3 = vaddq_f32(bias_neon_b, vmulq_n_f32(f32_3, scale[0]));
        f32_4 = vaddq_f32(bias_neon_g, vmulq_n_f32(f32_4, scale[1]));
        f32_5 = vaddq_f32(bias_neon_r, vmulq_n_f32(f32_5, scale[2]));

        int16x4_t s16_l0 = vqmovn_s32(VCVTAQ_S32_F32(f32_0));
        int16x8_t s16_0  = VQMOVN_HIGH_S32_T(s16_l0, VCVTAQ_S32_F32(f32_3));
        int16x4_t s16_l1 = vqmovn_s32(VCVTAQ_S32_F32(f32_1));
        int16x8_t s16_1  = VQMOVN_HIGH_S32_T(s16_l1, VCVTAQ_S32_F32(f32_4));
        int16x4_t s16_l2 = vqmovn_s32(VCVTAQ_S32_F32(f32_2));
        int16x8_t s16_2  = VQMOVN_HIGH_S32_T(s16_l2, VCVTAQ_S32_F32(f32_5));

        vi8x4.val[0] = vqmovn_s16(s16_0);
        vi8x4.val[1] = vqmovn_s16(s16_1);
        vi8x4.val[2] = vqmovn_s16(s16_2);

        vst4_s8(dst + i * 4, vi8x4);
    }
#endif
    for (; i < hw; ++i) {
        dst[4 * i + 0] = float2int8(scale[0] * src[3 * i + (reverse_channel ? 2 : 0)] + bias[0]);
        dst[4 * i + 1] = float2int8(scale[1] * src[3 * i + 1] + bias[1]);
        dst[4 * i + 2] = float2int8(scale[2] * src[3 * i + (reverse_channel ? 0 : 2)] + bias[2]);
        dst[4 * i + 3] = 0;
    }
}

template<typename T>
static void BGRToBlob(const uint8_t *src, T *dst, float *scale, float *bias, int hw,
                       bool reverse_channel) {
    if (reverse_channel) {
        BGRToBlobImpl<true>(src, dst, scale, bias, hw);
    } else {
        BGRToBlobImpl<false>(src, dst, scale, bias, hw);
    }
}

/*
convert data type from Tin to Tout, data format from nchw 2 nc4hw4
*/
template <typename Tin, typename Tout>
void NCHWToBlob(const Tin *src, Tout *dst, int channel, int hw, float *scale) {
    PackC4(dst, src, hw, channel);
}

/*
convert data type from float to int8, data format from nchw 2 nhwc
*/
template <>
void NCHWToBlob(const float *src, int8_t *dst, int channel, int hw, float *scale) {
    PackCAndQuant(dst, src, hw, channel, scale);
}

template <bool reverse_channel>
static void BlobToBGRAImpl(const float *src, uint8_t *dst, float *scale, float *bias, int hw, int channel) {
    int i = 0;
#ifdef TNN_USE_NEON
    float32x4_t bias_neon_b = vdupq_n_f32(bias[0]);
    float32x4_t bias_neon_g = vdupq_n_f32(bias[1]);
    float32x4_t bias_neon_r = vdupq_n_f32(bias[2]);
    float32x4_t bias_neon_a = vdupq_n_f32(bias[3]);
    uint8x8x4_t vi8x4;
    for (; i < hw - 7; i += 8) {
        float32x4x4_t vf32_0 = vld4q_f32(src + i * 4);
        float32x4x4_t vf32_1 = vld4q_f32(src + i * 4 + 16);

        vf32_0.val[0] = vaddq_f32(bias_neon_b, vmulq_n_f32(vf32_0.val[0], scale[0]));
        vf32_0.val[1] = vaddq_f32(bias_neon_g, vmulq_n_f32(vf32_0.val[1], scale[1]));
        vf32_0.val[2] = vaddq_f32(bias_neon_r, vmulq_n_f32(vf32_0.val[2], scale[2]));
        vf32_0.val[3] = vaddq_f32(bias_neon_a, vmulq_n_f32(vf32_0.val[3], scale[3]));
        vf32_1.val[0] = vaddq_f32(bias_neon_b, vmulq_n_f32(vf32_1.val[0], scale[0]));
        vf32_1.val[1] = vaddq_f32(bias_neon_g, vmulq_n_f32(vf32_1.val[1], scale[1]));
        vf32_1.val[2] = vaddq_f32(bias_neon_r, vmulq_n_f32(vf32_1.val[2], scale[2]));
        vf32_1.val[3] = vaddq_f32(bias_neon_a, vmulq_n_f32(vf32_1.val[3], scale[3]));

        int16x4_t s16_l0 = vqmovn_s32(VCVTAQ_S32_F32(vf32_0.val[reverse_channel ? 2 : 0]));
        int16x8_t s16_0  = VQMOVN_HIGH_S32_T(s16_l0, VCVTAQ_S32_F32(vf32_1.val[reverse_channel ? 2 : 0]));
        int16x4_t s16_l1 = vqmovn_s32(VCVTAQ_S32_F32(vf32_0.val[1]));
        int16x8_t s16_1  = VQMOVN_HIGH_S32_T(s16_l1, VCVTAQ_S32_F32(vf32_1.val[1]));
        int16x4_t s16_l2 = vqmovn_s32(VCVTAQ_S32_F32(vf32_0.val[reverse_channel ? 0 : 2]));
        int16x8_t s16_2  = VQMOVN_HIGH_S32_T(s16_l2, VCVTAQ_S32_F32(vf32_1.val[reverse_channel ? 0 : 2]));
        int16x4_t s16_l3 = vqmovn_s32(VCVTAQ_S32_F32(vf32_0.val[3]));
        int16x8_t s16_3  = VQMOVN_HIGH_S32_T(s16_l3, VCVTAQ_S32_F32(vf32_1.val[3]));

        vi8x4.val[0] = vqmovun_s16(s16_0);
        vi8x4.val[1] = vqmovun_s16(s16_1);
        vi8x4.val[2] = vqmovun_s16(s16_2);
        vi8x4.val[3] = vqmovun_s16(s16_3);

        if (channel == 3) {
            uint8x8x4_t vi8x4_tmp = vld4_u8(dst + i * 4);
            vi8x4.val[3]          = vi8x4_tmp.val[3];
        }

        vst4_u8(dst + i * 4, vi8x4);
    }
#endif
    for (; i < hw; ++i) {
        dst[4 * i + 0] = float2uint8(reverse_channel ? (scale[2] * src[4 * i + 2] + bias[2]) :
                                                       (scale[0] * src[4 * i + 0] + bias[0]));
        dst[4 * i + 1] = float2uint8(scale[1] * src[4 * i + 1] + bias[1]);
        dst[4 * i + 2] = float2uint8(reverse_channel ? (scale[0] * src[4 * i + 0] + bias[0]) :
                                                       (scale[2] * src[4 * i + 2] + bias[2]));
        if (channel == 4) {
            dst[4 * i + 3] = float2uint8(scale[3] * src[4 * i + 3] + bias[3]);
        }
    }
}

template <bool reverse_channel>
static void BlobToBGRAImpl(const int8_t *src, uint8_t *dst, float *scale, float *bias, int hw, int channel) {
    int i = 0;
#ifdef TNN_USE_NEON
    float32x4_t bias_neon_b = vdupq_n_f32(bias[0]);
    float32x4_t bias_neon_g = vdupq_n_f32(bias[1]);
    float32x4_t bias_neon_r = vdupq_n_f32(bias[2]);
    float32x4_t bias_neon_a = vdupq_n_f32(bias[3]);
    uint8x8x4_t vi8x4;
    for (; i < hw - 7; i += 8) {
        int8x8x4_t v_s8  = vld4_s8(src + i * 4);
        int16x8_t b_s16  = vmovl_s8(v_s8.val[0]);
        int16x8_t g_s16  = vmovl_s8(v_s8.val[1]);
        int16x8_t r_s16  = vmovl_s8(v_s8.val[2]);
        int16x8_t a_s16  = vmovl_s8(v_s8.val[3]);

        float32x4_t f32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_s16)));
        float32x4_t f32_1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(g_s16)));
        float32x4_t f32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(r_s16)));
        float32x4_t f32_3 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a_s16)));
        float32x4_t f32_4 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_s16)));
        float32x4_t f32_5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(g_s16)));
        float32x4_t f32_6 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(r_s16)));
        float32x4_t f32_7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a_s16)));

        f32_0 = vaddq_f32(bias_neon_b, vmulq_n_f32(f32_0, scale[0]));
        f32_1 = vaddq_f32(bias_neon_g, vmulq_n_f32(f32_1, scale[1]));
        f32_2 = vaddq_f32(bias_neon_r, vmulq_n_f32(f32_2, scale[2]));
        f32_3 = vaddq_f32(bias_neon_a, vmulq_n_f32(f32_3, scale[3]));
        f32_4 = vaddq_f32(bias_neon_b, vmulq_n_f32(f32_4, scale[0]));
        f32_5 = vaddq_f32(bias_neon_g, vmulq_n_f32(f32_5, scale[1]));
        f32_6 = vaddq_f32(bias_neon_r, vmulq_n_f32(f32_6, scale[2]));
        f32_7 = vaddq_f32(bias_neon_a, vmulq_n_f32(f32_7, scale[3]));

        int16x4_t s16_l0 = vqmovn_s32(VCVTAQ_S32_F32(reverse_channel ? f32_2 : f32_0));
        int16x8_t s16_0  = VQMOVN_HIGH_S32_T(s16_l0, VCVTAQ_S32_F32(reverse_channel ? f32_6 : f32_4));
        int16x4_t s16_l1 = vqmovn_s32(VCVTAQ_S32_F32(f32_1));
        int16x8_t s16_1  = VQMOVN_HIGH_S32_T(s16_l1, VCVTAQ_S32_F32(f32_5));
        int16x4_t s16_l2 = vqmovn_s32(VCVTAQ_S32_F32(reverse_channel ? f32_0 : f32_2));
        int16x8_t s16_2  = VQMOVN_HIGH_S32_T(s16_l2, VCVTAQ_S32_F32(reverse_channel ? f32_4 : f32_6));
        int16x4_t s16_l3 = vqmovn_s32(VCVTAQ_S32_F32(f32_3));
        int16x8_t s16_3  = VQMOVN_HIGH_S32_T(s16_l3, VCVTAQ_S32_F32(f32_7));

        vi8x4.val[0] = vqmovun_s16(s16_0);
        vi8x4.val[1] = vqmovun_s16(s16_1);
        vi8x4.val[2] = vqmovun_s16(s16_2);
        vi8x4.val[3] = vqmovun_s16(s16_3);

        if (channel == 3) {
            uint8x8x4_t vi8x4_tmp = vld4_u8(dst + i * 4);
            vi8x4.val[3]          = vi8x4_tmp.val[3];
        }

        vst4_u8(dst + i * 4, vi8x4);
    }
#endif
    for (; i < hw; ++i) {
        dst[4 * i + 0] = float2uint8(reverse_channel ? (scale[2] * src[4 * i + 2] + bias[2]) :
                                                       (scale[0] * src[4 * i + 0] + bias[0]));
        dst[4 * i + 1] = float2uint8(scale[1] * src[4 * i + 1] + bias[1]);
        dst[4 * i + 2] = float2uint8(reverse_channel ? (scale[0] * src[4 * i + 0] + bias[0]) :
                                                       (scale[2] * src[4 * i + 2] + bias[2]));
        if (channel == 4) {
            dst[4 * i + 3] = float2uint8(scale[3] * src[4 * i + 3] + bias[3]);
        }
    }
}

/*
if channel == 3, the fourth channel is ignored
*/
template<typename T>
static void BlobToBGRA(const T *src, uint8_t *dst, float *scale, float *bias, int hw,
                       bool reverse_channel, int channel) {
    if (reverse_channel) {
        BlobToBGRAImpl<true>(src, dst, scale, bias, hw, channel);
    } else {
        BlobToBGRAImpl<false>(src, dst, scale, bias, hw, channel);
    }
}

template <bool reverse_channel>
static void BlobToBGRImpl(const float *src, uint8_t *dst, float *scale, float *bias, int hw) {
    int i = 0;
#ifdef TNN_USE_NEON
    float32x4_t bias_neon_b = vdupq_n_f32(bias[0]);
    float32x4_t bias_neon_g = vdupq_n_f32(bias[1]);
    float32x4_t bias_neon_r = vdupq_n_f32(bias[2]);
    uint8x8x3_t vi8x3;
    for (; i < hw - 7; i += 8) {
        float32x4x4_t vf32_0 = vld4q_f32(src + i * 4);
        float32x4x4_t vf32_1 = vld4q_f32(src + i * 4 + 16);

        vf32_0.val[0] = vaddq_f32(bias_neon_b, vmulq_n_f32(vf32_0.val[0], scale[0]));
        vf32_0.val[1] = vaddq_f32(bias_neon_g, vmulq_n_f32(vf32_0.val[1], scale[1]));
        vf32_0.val[2] = vaddq_f32(bias_neon_r, vmulq_n_f32(vf32_0.val[2], scale[2]));
        vf32_1.val[0] = vaddq_f32(bias_neon_b, vmulq_n_f32(vf32_1.val[0], scale[0]));
        vf32_1.val[1] = vaddq_f32(bias_neon_g, vmulq_n_f32(vf32_1.val[1], scale[1]));
        vf32_1.val[2] = vaddq_f32(bias_neon_r, vmulq_n_f32(vf32_1.val[2], scale[2]));

        int16x4_t s16_l0 = vqmovn_s32(VCVTAQ_S32_F32(vf32_0.val[reverse_channel ? 2 : 0]));
        int16x8_t s16_0  = VQMOVN_HIGH_S32_T(s16_l0, VCVTAQ_S32_F32(vf32_1.val[reverse_channel ? 2 : 0]));
        int16x4_t s16_l1 = vqmovn_s32(VCVTAQ_S32_F32(vf32_0.val[1]));
        int16x8_t s16_1  = VQMOVN_HIGH_S32_T(s16_l1, VCVTAQ_S32_F32(vf32_1.val[1]));
        int16x4_t s16_l2 = vqmovn_s32(VCVTAQ_S32_F32(vf32_0.val[reverse_channel ? 0 : 2]));
        int16x8_t s16_2  = VQMOVN_HIGH_S32_T(s16_l2, VCVTAQ_S32_F32(vf32_1.val[reverse_channel ? 0 : 2]));

        vi8x3.val[0] = vqmovun_s16(s16_0);
        vi8x3.val[1] = vqmovun_s16(s16_1);
        vi8x3.val[2] = vqmovun_s16(s16_2);

        vst3_u8(dst + i * 3, vi8x3);
    }
#endif
    for (; i < hw; ++i) {
        dst[3 * i + 0] = float2uint8(reverse_channel ? (scale[2] * src[4 * i + 2] + bias[2]) :
                                                       (scale[0] * src[4 * i + 0] + bias[0]));
        dst[3 * i + 1] = float2uint8(scale[1] * src[4 * i + 1] + bias[1]);
        dst[3 * i + 2] = float2uint8(reverse_channel ? (scale[0] * src[4 * i + 0] + bias[0]) :
                                                       (scale[2] * src[4 * i + 2] + bias[2]));
    }
}

template <bool reverse_channel>
static void BlobToBGRImpl(const int8_t *src, uint8_t *dst, float *scale, float *bias, int hw) {
    int i = 0;
#ifdef TNN_USE_NEON
    float32x4_t bias_neon_b = vdupq_n_f32(bias[0]);
    float32x4_t bias_neon_g = vdupq_n_f32(bias[1]);
    float32x4_t bias_neon_r = vdupq_n_f32(bias[2]);
    uint8x8x3_t vi8x3;
    for (; i < hw - 7; i += 8) {
        int8x8x4_t v_s8  = vld4_s8(src + i * 4);
        int16x8_t b_s16  = vmovl_s8(v_s8.val[0]);
        int16x8_t g_s16  = vmovl_s8(v_s8.val[1]);
        int16x8_t r_s16  = vmovl_s8(v_s8.val[2]);

        float32x4_t f32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_s16)));
        float32x4_t f32_1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(g_s16)));
        float32x4_t f32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(r_s16)));
        float32x4_t f32_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_s16)));
        float32x4_t f32_4 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(g_s16)));
        float32x4_t f32_5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(r_s16)));

        f32_0 = vaddq_f32(bias_neon_b, vmulq_n_f32(f32_0, scale[0]));
        f32_1 = vaddq_f32(bias_neon_g, vmulq_n_f32(f32_1, scale[1]));
        f32_2 = vaddq_f32(bias_neon_r, vmulq_n_f32(f32_2, scale[2]));
        f32_3 = vaddq_f32(bias_neon_b, vmulq_n_f32(f32_3, scale[0]));
        f32_4 = vaddq_f32(bias_neon_g, vmulq_n_f32(f32_4, scale[1]));
        f32_5 = vaddq_f32(bias_neon_r, vmulq_n_f32(f32_5, scale[2]));

        int16x4_t s16_l0 = vqmovn_s32(VCVTAQ_S32_F32(reverse_channel ? f32_2 : f32_0));
        int16x8_t s16_0  = VQMOVN_HIGH_S32_T(s16_l0, VCVTAQ_S32_F32(reverse_channel ? f32_5 : f32_3));
        int16x4_t s16_l1 = vqmovn_s32(VCVTAQ_S32_F32(f32_1));
        int16x8_t s16_1  = VQMOVN_HIGH_S32_T(s16_l1, VCVTAQ_S32_F32(f32_4));
        int16x4_t s16_l2 = vqmovn_s32(VCVTAQ_S32_F32(reverse_channel ? f32_0 : f32_2));
        int16x8_t s16_2  = VQMOVN_HIGH_S32_T(s16_l2, VCVTAQ_S32_F32(reverse_channel ? f32_3 : f32_5));

        vi8x3.val[0] = vqmovun_s16(s16_0);
        vi8x3.val[1] = vqmovun_s16(s16_1);
        vi8x3.val[2] = vqmovun_s16(s16_2);

        vst3_u8(dst + i * 3, vi8x3);
    }
#endif
    for (; i < hw; ++i) {
        dst[3 * i + 0] = float2uint8(reverse_channel ? (scale[2] * src[4 * i + 2] + bias[2]) :
                                                       (scale[0] * src[4 * i + 0] + bias[0]));
        dst[3 * i + 1] = float2uint8(scale[1] * src[4 * i + 1] + bias[1]);
        dst[3 * i + 2] = float2uint8(reverse_channel ? (scale[0] * src[4 * i + 0] + bias[0]) :
                                                       (scale[2] * src[4 * i + 2] + bias[2]));
    }
}

template<typename T>
static void BlobToBGR(const T *src, uint8_t *dst, float *scale, float *bias, int hw,
                      bool reverse_channel) {
    if (reverse_channel) {
        BlobToBGRImpl<true>(src, dst, scale, bias, hw);
    } else {
        BlobToBGRImpl<false>(src, dst, scale, bias, hw);
    }
}

/*
reverse channel in format nchw
*/
template <typename T>
void NCHWChannelReverse(T *src, T *dst, int channel, int hw) {
    for (int c = 0; c < channel / 2; c++) {
        auto offset0 = c * hw;
        auto offset1 = (channel - 1 - c) * hw;
        std::vector<T> tmp(src + offset0, src + offset0 + hw);
        memcpy(dst + offset0, src + offset1, hw * sizeof(T));
        memcpy(dst + offset1, tmp.data(), hw * sizeof(T));
    }
}

/*
reverse channel in format nhwc
*/
template <typename T>
void NHWCChannelReverse(T *src, T *dst, int channel, int hw) {
    for (int i = 0; i < hw; i++) {
        for (int c = 0; c < channel / 2; c++) {
            T tmp                              = src[i * channel + c];
            dst[i * channel + c]               = src[i * channel + channel - 1 - c];
            dst[i * channel + channel - 1 - c] = tmp;
        }
    }
}

/*
reverse channel in format rgb uint8
*/
void RGBChannelReverse(uint8_t *src, uint8_t *dst, int channel, int hw) {
    int i = 0;
#ifdef TNN_USE_NEON
    for (; i + 15 < hw; i += 16) {
        uint8x16x3_t v_u8 = vld3q_u8(src + i * 3);
        uint8x16_t v_temp = v_u8.val[0];
        v_u8.val[0]       = v_u8.val[2];
        v_u8.val[2]       = v_temp;
        vst3q_u8(dst + i * 3, v_u8);
    }
#endif
    for (; i < hw; i++) {
        uint8_t tmp    = src[i * 3];
        dst[i * 3]     = src[i * 3 + 2];
        dst[i * 3 + 2] = tmp;
    }
}

/*
reverse channel in format rgba uint8, only reverse rgb
*/
void RGBAChannelReverse(uint8_t *src, uint8_t *dst, int channel, int hw) {
    int i = 0;
#ifdef TNN_USE_NEON
    for (; i + 15 < hw; i += 16) {
        uint8x16x4_t v_u8 = vld4q_u8(src + i * 4);
        uint8x16_t v_temp = v_u8.val[0];
        v_u8.val[0]       = v_u8.val[2];
        v_u8.val[2]       = v_temp;
        vst4q_u8(dst + i * 4, v_u8);
    }
#endif
    for (; i < hw; i++) {
        uint8_t tmp    = src[i * 4];
        dst[i * 4]     = src[i * 4 + 2];
        dst[i * 4 + 2] = tmp;
    }
}

template <typename T>
void ScaleBias(T *src, int channel, int hw, float *scale, float *bias, T *dst = nullptr) {
    if (dst == nullptr) {
        dst = src;
    }
    RawBuffer scale_buffer(ROUND_UP(channel, 4) * sizeof(float));
    RawBuffer bias_buffer(ROUND_UP(channel, 4) * sizeof(float));
    memcpy(scale_buffer.force_to<void *>(), scale, sizeof(float) * channel);
    memcpy(bias_buffer.force_to<void *>(), bias, sizeof(float) * channel);
    auto local_scale = scale_buffer.force_to<float *>();
    auto local_bias  = bias_buffer.force_to<float *>();

    for (int z = 0; z < UP_DIV(channel, 4); ++z) {
        auto src_z   = src + z * hw * 4;
        auto dst_z   = dst + z * hw * 4;
        auto v_scale = Float4::load(local_scale + z * 4);
        auto v_bias  = Float4::load(local_bias + z * 4);
        for (int s = 0; s < hw; ++s) {
            Float4::save(dst_z + s * 4, Float4::load(src_z + s * 4) * v_scale + v_bias);
        }
    }
}

void ArmBlobConverterAcc::ConvertImageToBlob(
        Mat& image, char *handle_ptr,
        const BlobDesc& desc, const DimsVector& dims, const int hw,
        MatConvertParam& param,
        std::vector<float>& fused_int8_scale,
        std::vector<float>& fused_int8_bias) {
    if (image.GetMatType() == N8UC4) {
        for (int n = 0; n < dims[0]; n++) {
            if (desc.data_type == DATA_TYPE_INT8) {
                BGRAToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw,
                           reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw, fused_int8_scale.data(),
                           fused_int8_bias.data(), hw, param.reverse_channel, dims[1]);
            } else {
                BGRAToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw,
                           reinterpret_cast<float *>(handle_ptr) + n * 4 * hw, param.scale.data(), param.bias.data(),
                           hw, param.reverse_channel, dims[1]);
            }
        }
    } else if (image.GetMatType() == N8UC3) {
        for (int n = 0; n < dims[0]; n++) {
            if (desc.data_type == DATA_TYPE_INT8) {
                BGRToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw,
                          reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw, fused_int8_scale.data(),
                          fused_int8_bias.data(), hw, param.reverse_channel);
            } else {
                BGRToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw,
                          reinterpret_cast<float *>(handle_ptr) + n * 4 * hw, param.scale.data(), param.bias.data(),
                          hw, param.reverse_channel);
            }
        }
    } else if (image.GetMatType() == NGRAY) {
        for (int n = 0; n < dims[0]; n++) {
            if (desc.data_type == DATA_TYPE_INT8) {
                GrayToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * hw,
                           reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw, fused_int8_scale[0], fused_int8_bias[0],
                           hw);
            } else {
                GrayToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * hw,
                           reinterpret_cast<float *>(handle_ptr) + n * 4 * hw, param.scale[0], param.bias[0], hw);
            }
        }
    }
}

void ArmBlobConverterAcc::ConvertYuvImageToBlob(
        Mat& image, char *handle_ptr,
        const BlobDesc& desc, const DimsVector& dims, const int hw,
        MatConvertParam& param,
        std::vector<float>& fused_int8_scale,
        std::vector<float>& fused_int8_bias) {
    if (image.GetMatType() == NNV12) {
        Mat bgr(DEVICE_ARM, N8UC3, image.GetDims());
        for (int n = 0; n < dims[0]; n++) {
            NV12ToBGR(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw / 2,
                      reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw, dims[2], dims[3]);
            if (desc.data_type == DATA_TYPE_INT8) {
                BGRToBlob(reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw,
                          reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw, fused_int8_scale.data(),
                          fused_int8_bias.data(), hw, param.reverse_channel);
            } else {
                BGRToBlob(reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw,
                          reinterpret_cast<float *>(handle_ptr) + n * 4 * hw, param.scale.data(), param.bias.data(),
                          hw, param.reverse_channel);
            }
        }
    } else if (image.GetMatType() == NNV21) {
        Mat bgr(DEVICE_ARM, N8UC3, image.GetDims());
        for (int n = 0; n < dims[0]; n++) {
            NV21ToBGR(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw / 2,
                      reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw, dims[2], dims[3]);
            if (desc.data_type == DATA_TYPE_INT8) {
                BGRToBlob(reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw,
                          reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw, fused_int8_scale.data(),
                          fused_int8_bias.data(), hw, param.reverse_channel);
            } else {
                BGRToBlob(reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw,
                          reinterpret_cast<float *>(handle_ptr) + n * 4 * hw, param.scale.data(), param.bias.data(),
                          hw, param.reverse_channel);
            }
        }
    }
}

Status ArmBlobConverterAcc::ConvertFloatMatToBlob(
        Mat& image, char *handle_ptr,
        const BlobDesc& desc, const DimsVector& dims, const int hw,
        const int c_r4,
        MatConvertParam& param,
        std::vector<float>& fused_int8_scale,
        std::vector<float>& fused_int8_bias) {
    if (desc.data_type == DATA_TYPE_INT8) {
        for (int n = 0; n < dims[0]; n++) {
            NCHWToBlob(reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw,
                        reinterpret_cast<int8_t *>(handle_ptr) + n * c_r4 * hw, dims[1], hw,
                        fused_int8_scale.data());
        }
    } else if (desc.data_type == DATA_TYPE_FLOAT) {
        for (int n = 0; n < dims[0]; n++) {
            NCHWToBlob(reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw,
                        reinterpret_cast<float *>(handle_ptr) + n * c_r4 * hw, dims[1], hw, nullptr);
            ScaleBias(reinterpret_cast<float *>(handle_ptr) + n * c_r4 * hw, dims[1], hw, param.scale.data(),
                        param.bias.data());
        }
    } else if (desc.data_type == DATA_TYPE_BFP16) {
        for (int n = 0; n < dims[0]; n++) {
            NCHWToBlob(reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw,
                        reinterpret_cast<bfp16_t *>(handle_ptr) + n * c_r4 * hw, dims[1], hw, nullptr);
            ScaleBias(reinterpret_cast<bfp16_t *>(handle_ptr) + n * c_r4 * hw, dims[1], hw, param.scale.data(),
                        param.bias.data());
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    return TNN_OK;
}

Status ArmBlobConverterAcc::ConvertBlobToFloatMat(
        Mat& image, char *handle_ptr,
        const DimsVector& dims, const int hw,
        const int c_r4, MatConvertParam& param,
        std::vector<float>& fused_int8_scale) {
    for (int n = 0; n < dims[0]; n++) {
        if (blob_->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            RawBuffer scale_biased(c_r4 * hw * sizeof(float));
            ScaleBias(reinterpret_cast<float *>(handle_ptr) + n * c_r4 * hw, dims[1], hw, param.scale.data(),
                        param.bias.data(), scale_biased.force_to<float *>());
            FloatBlobToNCHW(scale_biased.force_to<float *>(),
                            reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw, dims[1], hw);
        } else if (blob_->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
            RawBuffer scale_biased(c_r4 * hw * sizeof(bfp16_t));
            ScaleBias(reinterpret_cast<bfp16_t *>(handle_ptr) + n * c_r4 * hw, dims[1], hw, param.scale.data(),
                        param.bias.data(), scale_biased.force_to<bfp16_t *>());
            FloatBlobToNCHW(scale_biased.force_to<bfp16_t *>(),
                            reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw, dims[1], hw);
        } else if (blob_->GetBlobDesc().data_type == DATA_TYPE_INT8) {
            auto blob_int8 = reinterpret_cast<BlobInt8 *>(blob_);
            Int8BlobToNCHW(reinterpret_cast<int8_t *>(handle_ptr) + n * c_r4 * hw,
                            reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw, dims[1], hw,
                            fused_int8_scale.data(), fused_int8_bias.data());
        } else {
            return Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    }

    return TNN_OK;
}

Status ArmBlobConverterAcc::ConvertToMatAsync(Mat &image, MatConvertParam param, void *command_queue) {
    if (blob_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input/output blob is null");
    }
    auto desc       = blob_->GetBlobDesc();
    auto dims       = desc.dims;
    auto hw         = dims[2] * dims[3];
    auto c_r4       = ROUND_UP(dims[1], 4);
    auto handle_ptr = GetBlobHandlePtr(blob_->GetHandle());
    if (desc.data_type == DATA_TYPE_INT8) {
        if (fused_int8_scale.size() < c_r4) {
            fused_int8_scale.resize(c_r4);
            fused_int8_bias.resize(c_r4);
        }
        auto scale_handle = reinterpret_cast<BlobInt8 *>(blob_)->GetIntResource()->scale_handle;
        auto scale_data   = scale_handle.force_to<float *>();
        auto scale_count  = scale_handle.GetDataCount();
        for (int i = 0; i < dims[1]; i++) {
            auto scale_idx      = scale_count == 1 ? 0 : i;
            fused_int8_scale[i] = param.scale[i] * scale_data[scale_idx];
            fused_int8_bias[i]  = param.bias[i];
        }
    }

    if (image.GetMatType() == NCHW_FLOAT) {
        Status ret = ConvertBlobToFloatMat(image, handle_ptr, dims, hw, c_r4, param, fused_int8_scale);
        if (ret != TNN_OK) {
            return ret;
        }
    } else if (image.GetMatType() == RESERVED_BFP16_TEST && desc.data_type == DATA_TYPE_BFP16) {
        for (int n = 0; n < dims[0]; n++) {
            RawBuffer scale_biased(c_r4 * hw * sizeof(bfp16_t));
            ScaleBias(reinterpret_cast<bfp16_t *>(handle_ptr) + n * c_r4 * hw, dims[1], hw, param.scale.data(),
                      param.bias.data(), scale_biased.force_to<bfp16_t *>());
            FloatBlobToNCHW(scale_biased.force_to<bfp16_t *>(),
                            reinterpret_cast<bfp16_t *>(image.GetData()) + n * dims[1] * hw, dims[1], hw);
        }
    } else if (image.GetMatType() == RESERVED_INT8_TEST && desc.data_type == DATA_TYPE_INT8) {
        DataFormatConverter::ConvertFromNHWC4ToNCHWInt8(reinterpret_cast<int8_t *>(handle_ptr),
                                                        reinterpret_cast<int8_t *>(image.GetData()), dims[0], dims[1],
                                                        dims[2], dims[3]);
    } else if (image.GetMatType() == N8UC4) {
        for (int n = 0; n < dims[0]; n++) {
            if (desc.data_type == DATA_TYPE_INT8) {
                BlobToBGRA(reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw,
                           reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw, fused_int8_scale.data(),
                           fused_int8_bias.data(), hw, param.reverse_channel, dims[1]);
            } else {
                BlobToBGRA(reinterpret_cast<float *>(handle_ptr) + n * 4 * hw,
                           reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw, param.scale.data(), param.bias.data(),
                           hw, param.reverse_channel, dims[1]);
            }
        }
    } else if (image.GetMatType() == N8UC3) {
        for (int n = 0; n < dims[0]; n++) {
            if (desc.data_type == DATA_TYPE_INT8) {
                BlobToBGR(reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw,
                          reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw, fused_int8_scale.data(),
                          fused_int8_bias.data(), hw, param.reverse_channel);
            } else {
                BlobToBGR(reinterpret_cast<float *>(handle_ptr) + n * 4 * hw,
                          reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw, param.scale.data(), param.bias.data(),
                          hw, param.reverse_channel);
            }
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    return TNN_OK;
}

Status ArmBlobConverterAcc::ConvertFromMatAsync(Mat &image, MatConvertParam param, void *command_queue) {
    Status ret = TNN_OK;
    if (blob_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input/output blob_ is null");
    }
    auto desc       = blob_->GetBlobDesc();
    auto dims       = desc.dims;
    auto hw         = dims[2] * dims[3];
    auto handle_ptr = GetBlobHandlePtr(blob_->GetHandle());
    auto c_r4       = ROUND_UP(dims[1], 4);
    if (desc.data_type == DATA_TYPE_INT8) {
        if (fused_int8_scale.size() < c_r4) {
            fused_int8_scale.resize(c_r4);
            fused_int8_bias.resize(c_r4);
        }
        auto scale_handle = reinterpret_cast<BlobInt8 *>(blob_)->GetIntResource()->scale_handle;
        auto scale_data   = scale_handle.force_to<float *>();
        auto scale_count  = scale_handle.GetDataCount();
        for (int i = 0; i < dims[1]; i++) {
            auto scale_idx = scale_count == 1 ? 0 : i;
            if (scale_data[scale_idx] != 0) {
                fused_int8_scale[i] = param.scale[i] / scale_data[scale_idx];
                fused_int8_bias[i]  = param.bias[i] / scale_data[scale_idx];
            } else {
                fused_int8_scale[i] = 0;
                fused_int8_bias[i]  = 0;
            }
        }
    }

    if (image.GetMatType() == N8UC4 || image.GetMatType() == N8UC3 || image.GetMatType() == NGRAY) {
        ConvertImageToBlob(image, handle_ptr, desc, dims, hw, param, fused_int8_scale, fused_int8_bias);
    } else if (image.GetMatType() == NNV12 || image.GetMatType() == NNV21) {
        ConvertYuvImageToBlob(image, handle_ptr, desc, dims, hw, param, fused_int8_scale, fused_int8_bias);
    } else if (image.GetMatType() == NCHW_FLOAT) {
        ret = ConvertFloatMatToBlob(image, handle_ptr, desc, dims, hw, c_r4, param, fused_int8_scale, fused_int8_bias);
        if (ret != TNN_OK) {
            return ret;
        }
    } else if (image.GetMatType() == RESERVED_BFP16_TEST && desc.data_type == DATA_TYPE_BFP16) {
        for (int n = 0; n < dims[0]; n++) {
            NCHWToBlob(reinterpret_cast<bfp16_t *>(image.GetData()) + n * dims[1] * hw,
                       reinterpret_cast<bfp16_t *>(handle_ptr) + n * c_r4 * hw, dims[1], hw, nullptr);
            ScaleBias(reinterpret_cast<bfp16_t *>(handle_ptr) + n * c_r4 * hw, dims[1], hw, param.scale.data(),
                      param.bias.data());
        }
    } else if (image.GetMatType() == RESERVED_INT8_TEST && desc.data_type == DATA_TYPE_INT8) {
        DataFormatConverter::ConvertFromNCHWToNHWC4Int8(reinterpret_cast<int8_t *>(image.GetData()),
                                                        reinterpret_cast<int8_t *>(handle_ptr), dims[0], dims[1],
                                                        dims[2], dims[3]);
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    return ret;
}

/*
compatiable to ncnn mat
*/
Status ArmBlobConverterAcc::ConvertToMat(Mat &image, MatConvertParam param, void *command_queue) {
    return ConvertToMatAsync(image, param, command_queue);
}

/*
compatiable to ncnn mat
*/
Status ArmBlobConverterAcc::ConvertFromMat(Mat &image, MatConvertParam param, void *command_queue) {
    return ConvertFromMatAsync(image, param, command_queue);
}

DECLARE_BLOB_CONVERTER_CREATER(Arm);
REGISTER_BLOB_CONVERTER(Arm, DEVICE_ARM);

}  // namespace TNN_NS
