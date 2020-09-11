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
static void Int8BlobToNCHW(const int8_t *src, float *dst, int channel, int hw, float *scale) {
    UnpackAndDequant(dst, src, hw, channel, scale);
    return;
}

/*
convert data type from uint8 to float, data format from nhwc 2 nchw
*/
static void BGRAToBlob(const uint8_t *src, float *dst, float *scale, float *bias, int hw) {
    int i = 0;
#ifdef TNN_USE_NEON
    float32x4_t scale_neon = vld1q_f32(scale);
    float32x4_t bias_neon  = vld1q_f32(bias);
    for (; i < hw - 3; i += 4) {
        uint8x16_t v_u8    = vld1q_u8(src + i * 4);
        int16x8_t v_s16l   = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_u8)));
        int16x8_t v_s16h   = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_u8)));
        float32x4_t vf32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_s16l)));
        float32x4_t vf32_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_s16l)));
        float32x4_t vf32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_s16h)));
        float32x4_t vf32_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_s16h)));
        float32x4_t rf32_0 = vaddq_f32(bias_neon, vmulq_f32(scale_neon, vf32_0));
        float32x4_t rf32_1 = vaddq_f32(bias_neon, vmulq_f32(scale_neon, vf32_1));
        float32x4_t rf32_2 = vaddq_f32(bias_neon, vmulq_f32(scale_neon, vf32_2));
        float32x4_t rf32_3 = vaddq_f32(bias_neon, vmulq_f32(scale_neon, vf32_3));
        vst1q_f32(dst + i * 4 + 0, rf32_0);
        vst1q_f32(dst + i * 4 + 4, rf32_1);
        vst1q_f32(dst + i * 4 + 8, rf32_2);
        vst1q_f32(dst + i * 4 + 12, rf32_3);
    }
#endif
    for (; i < hw; ++i) {
        dst[4 * i + 0] = scale[0] * src[4 * i + 0] + bias[0];
        dst[4 * i + 1] = scale[1] * src[4 * i + 1] + bias[1];
        dst[4 * i + 2] = scale[2] * src[4 * i + 2] + bias[2];
        dst[4 * i + 3] = scale[3] * src[4 * i + 3] + bias[3];
    }
}

/*
convert data type from uint8 to float, data format from nhw4 2 nc4hw4
*/
static void BGRAToBlob(const uint8_t *src, int8_t *dst, float *scale, float *bias, int hw) {
    int i = 0;
#ifdef TNN_USE_NEON
    float32x4_t scale_neon = vld1q_f32(scale);
    float32x4_t bias_neon  = vld1q_f32(bias);
    for (; i < hw - 3; i += 4) {
        uint8x16_t v_u8    = vld1q_u8(src + i * 4);
        int16x8_t v_s16l   = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_u8)));
        int16x8_t v_s16h   = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_u8)));
        float32x4_t vf32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_s16l)));
        float32x4_t vf32_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_s16l)));
        float32x4_t vf32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_s16h)));
        float32x4_t vf32_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_s16h)));
        float32x4_t rf32_0 = vaddq_f32(bias_neon, vmulq_f32(scale_neon, vf32_0));
        float32x4_t rf32_1 = vaddq_f32(bias_neon, vmulq_f32(scale_neon, vf32_1));
        float32x4_t rf32_2 = vaddq_f32(bias_neon, vmulq_f32(scale_neon, vf32_2));
        float32x4_t rf32_3 = vaddq_f32(bias_neon, vmulq_f32(scale_neon, vf32_3));

        int16x4_t s16_00 = vqmovn_s32(VCVTAQ_S32_F32(rf32_0));
        int16x4_t s16_10 = vqmovn_s32(VCVTAQ_S32_F32(rf32_2));
        int16x8_t s16_0  = VQMOVN_HIGH_S32_T(s16_00, VCVTAQ_S32_F32(rf32_1));
        int16x8_t s16_1  = VQMOVN_HIGH_S32_T(s16_10, VCVTAQ_S32_F32(rf32_3));
        vst1_s8(dst + i * 4 + 0, vqmovn_s16(s16_0));
        vst1_s8(dst + i * 4 + 8, vqmovn_s16(s16_1));
    }
#endif
    for (; i < hw; ++i) {
        dst[4 * i + 0] = float2int8(scale[0] * src[4 * i + 0] + bias[0]);
        dst[4 * i + 1] = float2int8(scale[1] * src[4 * i + 1] + bias[1]);
        dst[4 * i + 2] = float2int8(scale[2] * src[4 * i + 2] + bias[2]);
        dst[4 * i + 3] = float2int8(scale[3] * src[4 * i + 3] + bias[3]);
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
static void BGRToBlob(const uint8_t *src, float *dst, float *scale, float *bias, int hw) {
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

        vf32.val[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_s16)));
        vf32.val[1] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(g_s16)));
        vf32.val[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(r_s16)));

        vf32.val[0] = vaddq_f32(bias_neon_b, vmulq_n_f32(vf32.val[0], scale[0]));
        vf32.val[1] = vaddq_f32(bias_neon_g, vmulq_n_f32(vf32.val[1], scale[1]));
        vf32.val[2] = vaddq_f32(bias_neon_r, vmulq_n_f32(vf32.val[2], scale[2]));

        vst4q_f32(dst + i * 4, vf32);

        vf32.val[0] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_s16)));
        vf32.val[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(g_s16)));
        vf32.val[2] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(r_s16)));

        vf32.val[0] = vaddq_f32(bias_neon_b, vmulq_n_f32(vf32.val[0], scale[0]));
        vf32.val[1] = vaddq_f32(bias_neon_g, vmulq_n_f32(vf32.val[1], scale[1]));
        vf32.val[2] = vaddq_f32(bias_neon_r, vmulq_n_f32(vf32.val[2], scale[2]));

        vst4q_f32(dst + i * 4 + 16, vf32);
    }
#endif
    for (; i < hw; ++i) {
        dst[4 * i + 0] = scale[0] * src[3 * i + 0] + bias[0];
        dst[4 * i + 1] = scale[1] * src[3 * i + 1] + bias[1];
        dst[4 * i + 2] = scale[2] * src[3 * i + 2] + bias[2];
        dst[4 * i + 3] = 0;
    }
}

/*
convert data type from uint8 to float, data format from nhw3 2 nc4hw4
*/
static void BGRToBlob(const uint8_t *src, int8_t *dst, float *scale, float *bias, int hw) {
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
        dst[4 * i + 0] = float2int8(scale[0] * src[3 * i + 0] + bias[0]);
        dst[4 * i + 1] = float2int8(scale[1] * src[3 * i + 1] + bias[1]);
        dst[4 * i + 2] = float2int8(scale[2] * src[3 * i + 2] + bias[2]);
        dst[4 * i + 3] = 0;
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
        }

        auto scale_handle = reinterpret_cast<BlobInt8 *>(blob_)->GetIntResource()->scale_handle;
        auto scale_data   = scale_handle.force_to<float *>();
        auto scale_count  = scale_handle.GetDataCount();

        for (int i = 0; i < dims[1]; i++) {
            auto scale_idx      = scale_count == 1 ? 0 : i;
            fused_int8_scale[i] = scale_data[scale_idx];
        }
    }
    if (image.GetMatType() == NCHW_FLOAT) {
        for (int n = 0; n < dims[0]; n++) {
            if (blob_->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
                FloatBlobToNCHW(reinterpret_cast<float *>(handle_ptr) + n * c_r4 * hw,
                                reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw, dims[1], hw);
            } else if (blob_->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
                FloatBlobToNCHW(reinterpret_cast<bfp16_t *>(handle_ptr) + n * c_r4 * hw,
                                reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw, dims[1], hw);
            } else if (blob_->GetBlobDesc().data_type == DATA_TYPE_INT8) {
                auto blob_int8 = reinterpret_cast<BlobInt8 *>(blob_);
                Int8BlobToNCHW(reinterpret_cast<int8_t *>(handle_ptr) + n * c_r4 * hw,
                               reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw, dims[1], hw,
                               fused_int8_scale.data());
            } else {
                return Status(TNNERR_PARAM_ERR, "convert type not support yet");
            }
        }
    } else if (image.GetMatType() == RESERVED_BFP16_TEST && desc.data_type == DATA_TYPE_BFP16) {
        for (int n = 0; n < dims[0]; n++) {
            FloatBlobToNCHW(reinterpret_cast<bfp16_t *>(handle_ptr) + n * c_r4 * hw,
                            reinterpret_cast<bfp16_t *>(image.GetData()) + n * dims[1] * hw, dims[1], hw);
        }
    } else if (image.GetMatType() == RESERVED_INT8_TEST && desc.data_type == DATA_TYPE_INT8) {
        DataFormatConverter::ConvertFromNHWC4ToNCHWInt8(reinterpret_cast<int8_t *>(handle_ptr),
                                                        reinterpret_cast<int8_t *>(image.GetData()), dims[0], dims[1],
                                                        dims[2], dims[3]);
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    // reverse channel after convert if needed
    if (param.reverse_channel) {
        if (image.GetMatType() == NCHW_FLOAT) {
            for (int n = 0; n < dims[0]; n++) {
                NCHWChannelReverse(reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw,
                                   reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw, dims[1], hw);
            }
        } else if (image.GetMatType() == RESERVED_BFP16_TEST && desc.data_type == DATA_TYPE_BFP16) {
            for (int n = 0; n < dims[0]; n++) {
                NCHWChannelReverse(reinterpret_cast<bfp16_t *>(image.GetData()) + n * dims[1] * hw,
                                   reinterpret_cast<bfp16_t *>(image.GetData()) + n * dims[1] * hw, dims[1], hw);
            }
        } else if (image.GetMatType() == RESERVED_INT8_TEST && desc.data_type == DATA_TYPE_INT8) {
            for (int n = 0; n < dims[0]; n++) {
                NCHWChannelReverse(reinterpret_cast<int8_t *>(image.GetData()) + n * dims[1] * hw,
                                   reinterpret_cast<int8_t *>(image.GetData()) + n * dims[1] * hw, dims[1], hw);
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "reverse type not support yet");
        }
    }

    return TNN_OK;
}

void ScaleBias(float *data, int channel, int hw, float *scale, float *bias) {
    for (int c = 0; c < channel; ++c) {
        int plane       = c / 4;
        int offset      = c % 4;
        auto *dataPlane = plane * hw * 4 + data;
        for (int s = 0; s < hw; ++s) {
            auto v            = dataPlane[offset];
            dataPlane[offset] = v * scale[c] + bias[c];
        }
    }
}

Status ArmBlobConverterAcc::ConvertFromMatAsync(Mat &image_src, MatConvertParam param, void *command_queue) {
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
        auto blob_scale = reinterpret_cast<BlobInt8 *>(blob_)->GetIntResource()->scale_handle.force_to<float *>();
        for (int i = 0; i < dims[1]; i++) {
            if (blob_scale[i] != 0) {
                fused_int8_scale[i] = param.scale[i] / blob_scale[i];
                fused_int8_bias[i]  = param.bias[i] / blob_scale[i];
            } else {
                fused_int8_scale[i] = 0;
                fused_int8_bias[i]  = 0;
            }
        }
    }

    Mat image(image_src.GetDeviceType(), image_src.GetMatType(), image_src.GetDims(), image_src.GetData());

    // reverse channel before convert if needed
    if (param.reverse_channel) {
        Mat reversed(image.GetDeviceType(), image.GetMatType(), image.GetDims());
        if (image.GetMatType() == N8UC3) {
            for (int n = 0; n < dims[0]; n++) {
                RGBChannelReverse(reinterpret_cast<uint8_t *>(image.GetData()) + n * dims[1] * hw,
                                  reinterpret_cast<uint8_t *>(reversed.GetData()) + n * dims[1] * hw, dims[1], hw);
            }
        } else if (image.GetMatType() == N8UC4) {
            for (int n = 0; n < dims[0]; n++) {
                RGBAChannelReverse(reinterpret_cast<uint8_t *>(image.GetData()) + n * dims[1] * hw,
                                   reinterpret_cast<uint8_t *>(reversed.GetData()) + n * dims[1] * hw, dims[1], hw);
            }
        } else if (image.GetMatType() == NCHW_FLOAT) {
            for (int n = 0; n < dims[0]; n++) {
                NCHWChannelReverse(reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw,
                                   reinterpret_cast<float *>(reversed.GetData()) + n * dims[1] * hw, dims[1], hw);
            }
        } else if (image.GetMatType() == RESERVED_BFP16_TEST && desc.data_type == DATA_TYPE_BFP16) {
            for (int n = 0; n < dims[0]; n++) {
                NCHWChannelReverse(reinterpret_cast<bfp16_t *>(image.GetData()) + n * dims[1] * hw,
                                   reinterpret_cast<bfp16_t *>(reversed.GetData()) + n * dims[1] * hw, dims[1], hw);
            }
        } else if (image.GetMatType() == RESERVED_INT8_TEST && desc.data_type == DATA_TYPE_INT8) {
            for (int n = 0; n < dims[0]; n++) {
                NCHWChannelReverse(reinterpret_cast<int8_t *>(image.GetData()) + n * dims[1] * hw,
                                   reinterpret_cast<int8_t *>(reversed.GetData()) + n * dims[1] * hw, dims[1], hw);
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "reverse type not support yet");
        }
        image = reversed;
    }

    if (image.GetMatType() == N8UC4) {
        for (int n = 0; n < dims[0]; n++) {
            if (desc.data_type == DATA_TYPE_INT8) {
                BGRAToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw,
                           reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw, fused_int8_scale.data(),
                           fused_int8_bias.data(), hw);
            } else {
                BGRAToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw,
                           reinterpret_cast<float *>(handle_ptr) + n * 4 * hw, param.scale.data(), param.bias.data(),
                           hw);
            }
        }
    } else if (image.GetMatType() == N8UC3) {
        for (int n = 0; n < dims[0]; n++) {
            if (desc.data_type == DATA_TYPE_INT8) {
                BGRToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw,
                          reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw, fused_int8_scale.data(),
                          fused_int8_bias.data(), hw);
            } else {
                BGRToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw,
                          reinterpret_cast<float *>(handle_ptr) + n * 4 * hw, param.scale.data(), param.bias.data(),
                          hw);
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
    } else if (image.GetMatType() == NNV12) {
        Mat bgr(DEVICE_ARM, N8UC3, image.GetDims());
        for (int n = 0; n < dims[0]; n++) {
            NV12ToBGR(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw / 2,
                      reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw, dims[2], dims[3]);
            if (desc.data_type == DATA_TYPE_INT8) {
                BGRToBlob(reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw,
                          reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw, fused_int8_scale.data(),
                          fused_int8_bias.data(), hw);
            } else {
                BGRToBlob(reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw,
                          reinterpret_cast<float *>(handle_ptr) + n * 4 * hw, param.scale.data(), param.bias.data(),
                          hw);
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
                          fused_int8_bias.data(), hw);
            } else {
                BGRToBlob(reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw,
                          reinterpret_cast<float *>(handle_ptr) + n * 4 * hw, param.scale.data(), param.bias.data(),
                          hw);
            }
        }
    } else if (image.GetMatType() == NCHW_FLOAT) {
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
                if (dims[0] == 1 && dims[1] == 2 && hw == 81) {
                    auto ptr = reinterpret_cast<float *>(handle_ptr);
                    for (int i = 0; i < hw; i++) {
                        printf("Init %f %f %f %f\n", ptr[i * 4 + 0], ptr[i * 4 + 1], ptr[i * 4 + 2], ptr[i * 4 + 3]);
                    }
                }
                // devandong
                ScaleBias(reinterpret_cast<float *>(handle_ptr) + n * c_r4 * hw, dims[1], hw, param.scale.data(),
                          param.bias.data());
            }
        } else if (desc.data_type == DATA_TYPE_BFP16) {
            for (int n = 0; n < dims[0]; n++) {
                NCHWToBlob(reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw,
                           reinterpret_cast<bfp16_t *>(handle_ptr) + n * c_r4 * hw, dims[1], hw, nullptr);
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "convert type not support yet");
        }
    } else if (image.GetMatType() == RESERVED_BFP16_TEST && desc.data_type == DATA_TYPE_BFP16) {
        for (int n = 0; n < dims[0]; n++) {
            NCHWToBlob(reinterpret_cast<bfp16_t *>(image.GetData()) + n * dims[1] * hw,
                       reinterpret_cast<bfp16_t *>(handle_ptr) + n * c_r4 * hw, dims[1], hw, nullptr);
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
