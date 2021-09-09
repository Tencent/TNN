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
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

ArmBlobConverterAcc::ArmBlobConverterAcc(Blob *blob) : BlobConverterAcc(blob) {}
ArmBlobConverterAcc::~ArmBlobConverterAcc() {}

std::string ArmBlobConverterAcc::GetUniqueBlobConvertKey(MatType mat_type, DataType data_type,
                                                         BlobConvertDirection cvt_dir) {
    return ToString(mat_type) + "_" + ToString(data_type) + "_" + ToString(cvt_dir);
}

std::map<std::string, ArmBlobConvertFunc>& ArmBlobConverterAcc::GetBlobConvertFuncMap() {
    static std::map<std::string, ArmBlobConvertFunc> cvt_map;
    return cvt_map;
}

Status ArmBlobConverterAcc::RegisterBlobConvertFunc(MatType mat_type, DataType data_type,
                                                    BlobConvertDirection cvt_dir, ArmBlobConvertFunc cvt_func) {
    auto& cvt_map       = GetBlobConvertFuncMap();
    const auto& cvt_key = GetUniqueBlobConvertKey(mat_type, data_type, cvt_dir);
    cvt_map[cvt_key] = cvt_func;
    return TNN_OK;
}

Status ArmBlobConverterAcc::GetBlobConvertFunc(MatType mat_type, DataType data_type,
                                               BlobConvertDirection cvt_dir, ArmBlobConvertFunc& cvt_func) {
    const auto& cvt_map = GetBlobConvertFuncMap();
    const auto& cvt_key = GetUniqueBlobConvertKey(mat_type, data_type, cvt_dir);
    if (cvt_map.find(cvt_key) == cvt_map.end() || cvt_map.at(cvt_key) == nullptr) {
        LOGE("ArmBlobConverterAcc::GetBlobConvertFunc, convert type not support yet. mat_type: %d data_type:%d cvt_dir:%d\n", mat_type, data_type, cvt_dir);
#if !TNN_ARM82
        if (data_type == DATA_TYPE_HALF) {
            LOGE("ArmBlobConverterAcc::GetBlobConvertFunc, fp16 is used while TNN_ARM82 is off, try to open TNN_ARM82 and run again.\n");
        }
#endif
        return Status(TNNERR_PARAM_ERR, "ArmBlobConverterAcc::GetBlobConvertFunc, convert type not support yet");
    }
    cvt_func = cvt_map.at(cvt_key);
    return TNN_OK;
}

Status ArmBlobConverterAcc::ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue) {
    Status ret = TNN_OK;
    if (blob_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input/output blob is null");
    }
    auto desc       = blob_->GetBlobDesc();
    auto dims       = desc.dims;
    auto batch      = DimsFunctionUtils::GetDim(dims, 0);
    auto channel    = DimsFunctionUtils::GetDim(dims, 1);
    auto hw         = DimsVectorUtils::Count(dims, 2);
    auto c_r4       = ROUND_UP(channel, 4);
    auto handle_ptr = GetBlobHandlePtr(blob_->GetHandle());
    if (desc.data_type == DATA_TYPE_INT8) {
        if (fused_int8_scale.size() < c_r4) {
            fused_int8_scale.resize(c_r4);
            fused_int8_bias.resize(c_r4);
        }
        auto scale_handle = reinterpret_cast<BlobInt8*>(blob_)->GetIntResource()->scale_handle;
        auto scale_data   = scale_handle.force_to<float*>();
        auto scale_count  = scale_handle.GetDataCount();
        for (int i = 0; i < channel; i++) {
            auto scale_idx      = scale_count == 1 ? 0 : i;
            fused_int8_scale[i] = param.scale[i] * scale_data[scale_idx];
            fused_int8_bias[i]  = param.bias[i];
        }
    } else if (desc.data_type == DATA_TYPE_INT32) {
        int count    = DimsVectorUtils::Count(blob_->GetBlobDesc().dims);
        int ele_size = DataTypeUtils::GetBytesSize(desc.data_type);
        if (image.GetMatType() == NC_INT32) {
            memcpy(image.GetData(), GetBlobHandlePtr(blob_->GetHandle()), count * ele_size);
        }
        return ret;
    }

    auto cvt_data_type  = desc.data_type;
    auto cvt_handle_ptr = handle_ptr;

    // pack if data format is nchw
    RawBuffer tmp_packed_blob;
    if (desc.data_format == DATA_FORMAT_NCHW) {
        if (desc.data_type == DATA_TYPE_FLOAT) {
            tmp_packed_blob = RawBuffer(batch * c_r4 * hw * DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT));
            auto dst_ptr    = tmp_packed_blob.force_to<float*>();
            auto src_ptr    = reinterpret_cast<float*>(cvt_handle_ptr);
            for (int n = 0; n < batch; ++n) {
                auto dst_ptr_n = dst_ptr + n * c_r4 * hw;
                auto src_ptr_n = src_ptr + n * channel * hw;
                PackC4(dst_ptr_n, src_ptr_n, hw, channel);
            }
        }
#if TNN_ARM82
        else if (desc.data_type == DATA_TYPE_HALF) {
            tmp_packed_blob = RawBuffer(batch * ROUND_UP(c_r4, 8) * hw * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
            auto dst_ptr    = tmp_packed_blob.force_to<fp16_t*>();
            auto src_ptr    = reinterpret_cast<fp16_t*>(cvt_handle_ptr);
            for (int n = 0; n < batch; ++n) {
                auto dst_ptr_n = dst_ptr + n * ROUND_UP(c_r4, 8) * hw;
                auto src_ptr_n = src_ptr + n * channel * hw;
                PackC8(dst_ptr_n, src_ptr_n, hw, channel);
            }
        }
#endif
        else {
            LOGE("ArmBlobConverterAcc::ConvertToMatAsync, not support data type for nchw blob, %d\n", desc.data_type);
            return Status(TNNERR_PARAM_ERR,
                          "ArmBlobConverterAcc::ConvertToMatAsync not support data type for nchw blob");
        }
        cvt_handle_ptr = tmp_packed_blob.force_to<char*>();
    }

#ifdef TNN_ARM82_A32
    RawBuffer tmp_float_blob;
    if (desc.data_type == DATA_TYPE_HALF) {
        // In aarch32 or armv7, first reformat half blob to float blob.
        tmp_float_blob = RawBuffer(batch * c_r4 * hw * DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT));
        HalfC8ToFloatC4(tmp_float_blob.force_to<float*>(), reinterpret_cast<fp16_t*>(cvt_handle_ptr), batch, channel,
                        DimsVectorUtils::Count(dims, 2));
        // In aarch32 or armv7, then convert from float blob.
        cvt_data_type  = DATA_TYPE_FLOAT;
        cvt_handle_ptr = tmp_float_blob.force_to<char*>();
    }
#endif

    ret = GetBlobConvertFunc(image.GetMatType(), cvt_data_type, CVT_DIR_BLOB2MAT, cvt_func_);
    if (ret == TNN_OK) {
        return cvt_func_(image, cvt_handle_ptr, param, dims, hw, c_r4, fused_int8_scale, fused_int8_bias);
    } else {
        return ret;
    }
}

Status ArmBlobConverterAcc::ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue) {
    Status ret = TNN_OK;
    if (blob_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input/output blob_ is null");
    }
    auto desc       = blob_->GetBlobDesc();
    auto dims       = desc.dims;
    auto batch      = DimsFunctionUtils::GetDim(dims, 0);
    auto channel    = DimsFunctionUtils::GetDim(dims, 1);
    auto hw         = DimsVectorUtils::Count(dims, 2);
    auto handle_ptr = GetBlobHandlePtr(blob_->GetHandle());
    auto c_r4       = ROUND_UP(channel, 4);
    if (desc.data_type == DATA_TYPE_INT8 && image.GetMatType() != RESERVED_INT8_TEST) {
        if (fused_int8_scale.size() < c_r4) {
            fused_int8_scale.resize(c_r4);
            fused_int8_bias.resize(c_r4);
        }
        auto int8_blob = dynamic_cast<BlobInt8*>(blob_);
        if (int8_blob == nullptr) {
            LOGE("TNN does not support the mat type: %d, please check you mat type\n", image.GetMatType());
            return Status(TNNERR_PARAM_ERR, "TNN does not support the mat type, please check you mat type");
        }
        auto scale_handle = int8_blob->GetIntResource()->scale_handle;
        auto scale_data   = scale_handle.force_to<float*>();
        auto scale_count  = scale_handle.GetDataCount();
        for (int i = 0; i < channel; i++) {
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

    auto cvt_data_type  = desc.data_type;
    auto cvt_handle_ptr = handle_ptr;

    // frist convert to packed buffer if data format is nchw
    RawBuffer tmp_packed_blob;
    if (desc.data_format == DATA_FORMAT_NCHW) {
        if (desc.data_type == DATA_TYPE_FLOAT) {
            tmp_packed_blob = RawBuffer(batch * c_r4 * hw * DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT));
        }
#if TNN_ARM82
        else if (desc.data_type == DATA_TYPE_HALF) {
            tmp_packed_blob = RawBuffer(batch * ROUND_UP(c_r4, 8) * hw * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
        }
#endif
        else {
            LOGE("ArmBlobConverterAcc::ConvertFromMatAsync, not support data type for nchw blob, %d\n", desc.data_type);
            return Status(TNNERR_PARAM_ERR,
                          "ArmBlobConverterAcc::ConvertFromMatAsync not support data type for nchw blob");
        }
        cvt_handle_ptr = tmp_packed_blob.force_to<char*>();
    }

    ret = GetBlobConvertFunc(image.GetMatType(), cvt_data_type, CVT_DIR_MAT2BLOB, cvt_func_);
    if (ret == TNN_OK) {
        ret = cvt_func_(image, cvt_handle_ptr, param, dims, hw, c_r4, fused_int8_scale, fused_int8_bias);
    } else {
        return ret;
    }

    // then unpack if data format is nchw
    if (desc.data_format == DATA_FORMAT_NCHW) {
        if (desc.data_type == DATA_TYPE_FLOAT) {
            auto dst_ptr = reinterpret_cast<float*>(handle_ptr);
            auto src_ptr = reinterpret_cast<float*>(cvt_handle_ptr);
            for (int n = 0; n < batch; ++n) {
                auto dst_ptr_n = dst_ptr + n * channel * hw;
                auto src_ptr_n = src_ptr + n * c_r4 * hw;
                UnpackC4(dst_ptr_n, src_ptr_n, hw, channel);
            }
        }
#if TNN_ARM82
        else if (desc.data_type == DATA_TYPE_HALF) {
            auto dst_ptr = reinterpret_cast<fp16_t*>(handle_ptr);
            auto src_ptr = reinterpret_cast<fp16_t*>(cvt_handle_ptr);
            for (int n = 0; n < batch; ++n) {
                auto dst_ptr_n = dst_ptr + n * channel * hw;
                auto src_ptr_n = src_ptr + n * ROUND_UP(c_r4, 8) * hw;
                UnpackC8(dst_ptr_n, src_ptr_n, hw, channel);
            }
        }
#endif
    }

    return ret;
}

/*
compatible to ncnn mat
*/
Status ArmBlobConverterAcc::ConvertToMat(Mat &image, MatConvertParam param, void *command_queue) {
    return ConvertToMatAsync(image, param, command_queue);
}

/*
compatible to ncnn mat
*/
Status ArmBlobConverterAcc::ConvertFromMat(Mat &image, MatConvertParam param, void *command_queue) {
    return ConvertFromMatAsync(image, param, command_queue);
}

DECLARE_BLOB_CONVERTER_CREATER(Arm);
REGISTER_BLOB_CONVERTER(Arm, DEVICE_ARM);

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

template <typename Tin, typename Tout>
void HalfBlobToNCHW(const Tin *src, Tout *dst, int channel, int hw) {
    if (channel % 4 == 0 && hw == 1 && sizeof(Tin) == sizeof(Tout)) {
        memcpy(dst, src, channel * sizeof(Tin));
        return;
    }
    UnpackC8(dst, src, hw, channel);
}

template void HalfBlobToNCHW(const fp16_t* src, float *dst, int channel, int hw);

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
static void BGRAToBlobImpl(const uint8_t *src, float *dst, const float *scale, const float *bias,
                           int hw, int channel) {
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
static void BGRAToBlobImpl(const uint8_t *src, int8_t *dst, const float *scale, const float *bias,
                           int hw, int channel) {
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
void BGRAToBlob(const uint8_t *src, T *dst, const float *scale, const float *bias, int hw,
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
static void GrayToBlob(const uint8_t *src, float *dst, const float scale, const float bias, int hw) {
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

        vst1q_lane_f32(dst + (i + 0) * 4, rf32_0, 0);
        vst1q_lane_f32(dst + (i + 1) * 4, rf32_0, 1);
        vst1q_lane_f32(dst + (i + 2) * 4, rf32_0, 2);
        vst1q_lane_f32(dst + (i + 3) * 4, rf32_0, 3);
        vst1q_lane_f32(dst + (i + 4) * 4, rf32_1, 0);
        vst1q_lane_f32(dst + (i + 5) * 4, rf32_1, 1);
        vst1q_lane_f32(dst + (i + 6) * 4, rf32_1, 2);
        vst1q_lane_f32(dst + (i + 7) * 4, rf32_1, 3);
    }
#endif
    for (; i < hw; ++i) {
        dst[4 * i] = scale * src[i] + bias;
    }
}

/*
convert data type from uint8 to int8, data format from nhw1 2 nhwc
*/
static void GrayToBlob(const uint8_t *src, int8_t *dst, const float scale, const float bias, int hw) {
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
static void BGRToBlobImpl(const uint8_t *src, float *dst, const float *scale, const float *bias, int hw) {
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
static void BGRToBlobImpl(const uint8_t *src, int8_t *dst, const float *scale, const float *bias, int hw) {
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
void BGRToBlob(const uint8_t *src, T *dst, const float *scale, const float *bias, int hw,
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

template <>
void NCHWToBlob(const fp16_t *src, fp16_t *dst, int channel, int hw, float *scale) {
    PackC8(dst, src, hw, channel);
}

template <>
void NCHWToBlob(const float *src, fp16_t *dst, int channel, int hw, float *scale) {
    PackC8(dst, src, hw, channel);
}

/*
convert data type from float to int8, data format from nchw 2 nhwc
*/
template <>
void NCHWToBlob(const float *src, int8_t *dst, int channel, int hw, float *scale) {
    PackCAndQuant(dst, src, hw, channel, scale);
}

template <bool reverse_channel>
static void BlobToBGRAImpl(const float *src, uint8_t *dst, const float *scale, const float *bias,
                           int hw, int channel) {
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
static void BlobToBGRAImpl(const int8_t *src, uint8_t *dst, const float *scale, const float *bias,
                           int hw, int channel) {
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
static void BlobToBGRA(const T *src, uint8_t *dst, const float *scale, const float *bias, int hw,
                       bool reverse_channel, int channel) {
    if (reverse_channel) {
        BlobToBGRAImpl<true>(src, dst, scale, bias, hw, channel);
    } else {
        BlobToBGRAImpl<false>(src, dst, scale, bias, hw, channel);
    }
}

template <bool reverse_channel>
static void BlobToBGRImpl(const float *src, uint8_t *dst, const float *scale, const float *bias, int hw) {
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
static void BlobToBGRImpl(const int8_t *src, uint8_t *dst, const float *scale, const float *bias, int hw) {
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
static void BlobToBGR(const T *src, uint8_t *dst, const float *scale, const float *bias, int hw,
                      bool reverse_channel) {
    if (reverse_channel) {
        BlobToBGRImpl<true>(src, dst, scale, bias, hw);
    } else {
        BlobToBGRImpl<false>(src, dst, scale, bias, hw);
    }
}

static Mat GetBGRFromYUV(Mat& image, const DimsVector& dims, const int hw, bool is_nv12) {
    Mat bgr(DEVICE_ARM, N8UC3, image.GetDims());
    auto batch = DimsFunctionUtils::GetDim(dims, 0);
    for (int n = 0; n < batch; n++) {
        if (is_nv12) {
            NV12ToBGR(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw / 2,
                      reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw, dims[2], dims[3]);
        } else {
            NV21ToBGR(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw / 2,
                      reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw, dims[2], dims[3]);
        }
    }
    return bgr;
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

bool NeedDoScaleBias(const MatConvertParam &param) {
    for (auto s : param.scale) {
        if (s != 1.0f) {
            return true;
        }
    }
    for (auto b : param.bias) {
        if (b != 0.0f) {
            return true;
        }
    }

    return false;
}

static Status ConvertN8UC4ToInt8Blob(Mat& image, char* handle_ptr, const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4, std::vector<float>& fused_int8_scale,
                                     std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        BGRAToBlob(reinterpret_cast<uint8_t*>(image.GetData()) + n * 4 * hw,
                   reinterpret_cast<int8_t*>(handle_ptr) + n * 4 * hw, fused_int8_scale.data(), fused_int8_bias.data(),
                   hw, param.reverse_channel, channel);
    }
    return TNN_OK;
}

static Status ConvertN8UC4ToFloatBlob(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                      const DimsVector& dims, const int hw, const int c_r4,
                                      std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        BGRAToBlob(reinterpret_cast<uint8_t*>(image.GetData()) + n * 4 * hw,
                   reinterpret_cast<float*>(handle_ptr) + n * 4 * hw, param.scale.data(), param.bias.data(), hw,
                   param.reverse_channel, channel);
    }
    return TNN_OK;
}

static Status ConvertN8UC3ToInt8Blob(Mat& image, char* handle_ptr, const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4, std::vector<float>& fused_int8_scale,
                                     std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        BGRToBlob(reinterpret_cast<uint8_t*>(image.GetData()) + n * 3 * hw,
                  reinterpret_cast<int8_t*>(handle_ptr) + n * 4 * hw, fused_int8_scale.data(), fused_int8_bias.data(),
                  hw, param.reverse_channel);
    }
    return TNN_OK;
}

static Status ConvertN8UC3ToFloatBlob(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                      const DimsVector& dims, const int hw, const int c_r4,
                                      std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        BGRToBlob(reinterpret_cast<uint8_t*>(image.GetData()) + n * 3 * hw,
                  reinterpret_cast<float*>(handle_ptr) + n * 4 * hw, param.scale.data(), param.bias.data(), hw,
                  param.reverse_channel);
    }
    return TNN_OK;
}

static Status ConvertNGRAYToInt8Blob(Mat& image, char* handle_ptr, const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4, std::vector<float>& fused_int8_scale,
                                     std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        GrayToBlob(reinterpret_cast<uint8_t*>(image.GetData()) + n * 1 * hw,
                   reinterpret_cast<int8_t*>(handle_ptr) + n * 4 * hw, fused_int8_scale[0], fused_int8_bias[0], hw);
    }
    return TNN_OK;
}

static Status ConvertNGRAYToFloatBlob(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                      const DimsVector& dims, const int hw, const int c_r4,
                                      std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        GrayToBlob(reinterpret_cast<uint8_t*>(image.GetData()) + n * 1 * hw,
                   reinterpret_cast<float*>(handle_ptr) + n * 4 * hw, param.scale[0], param.bias[0], hw);
    }
    return TNN_OK;
}

static Status ConvertNNV12ToInt8Blob(Mat& image, char* handle_ptr,
                                     const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4,
                                     std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    Mat bgr = GetBGRFromYUV(image, dims, hw, true);
    return ConvertN8UC3ToInt8Blob(bgr, handle_ptr, param, dims, hw, c_r4, fused_int8_scale, fused_int8_bias);
}

static Status ConvertNNV12ToFloatBlob(Mat& image, char* handle_ptr,
                                      const MatConvertParam& param, const DimsVector& dims,
                                      const int hw, const int c_r4,
                                      std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    Mat bgr = GetBGRFromYUV(image, dims, hw, true);
    return ConvertN8UC3ToFloatBlob(bgr, handle_ptr, param, dims, hw, c_r4, fused_int8_scale, fused_int8_bias);
}

static Status ConvertNNV21ToInt8Blob(Mat& image, char* handle_ptr,
                                     const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4,
                                     std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    Mat bgr = GetBGRFromYUV(image, dims, hw, false);
    return ConvertN8UC3ToInt8Blob(bgr, handle_ptr, param, dims, hw, c_r4, fused_int8_scale, fused_int8_bias);
}

static Status ConvertNNV21ToFloatBlob(Mat& image, char* handle_ptr,
                                      const MatConvertParam& param, const DimsVector& dims,
                                      const int hw, const int c_r4,
                                      std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    Mat bgr = GetBGRFromYUV(image, dims, hw, false);
    return ConvertN8UC3ToFloatBlob(bgr, handle_ptr, param, dims, hw, c_r4, fused_int8_scale, fused_int8_bias);
}

static Status ConvertNCHWFloatToInt8Blob(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                         const DimsVector& dims, const int hw, const int c_r4,
                                         std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        NCHWToBlob(reinterpret_cast<float*>(image.GetData()) + n * channel * hw,
                   reinterpret_cast<int8_t*>(handle_ptr) + n * c_r4 * hw, channel, hw, fused_int8_scale.data());
    }
    return TNN_OK;
}

template <typename T_mat, typename T_blob>
static Status ConvertFloatMatToFloatBlob(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                         const DimsVector& dims, const int hw, const int c_r4,
                                         std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    if (NeedDoScaleBias(param)) {
        for (int n = 0; n < batch; n++) {
            NCHWToBlob(reinterpret_cast<T_mat*>(image.GetData()) + n * channel * hw,
                       reinterpret_cast<T_blob*>(handle_ptr) + n * c_r4 * hw, channel, hw, nullptr);
            ScaleBias(reinterpret_cast<T_blob*>(handle_ptr) + n * c_r4 * hw, channel, hw, param.scale.data(),
                      param.bias.data());
        }
    } else {
        for (int n = 0; n < batch; n++) {
            NCHWToBlob(reinterpret_cast<T_mat*>(image.GetData()) + n * channel * hw,
                       reinterpret_cast<T_blob*>(handle_ptr) + n * c_r4 * hw, channel, hw, nullptr);
        }
    }
    return TNN_OK;
}

static Status ConvertInt8MatToInt8Blob(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                       const DimsVector& dims, const int hw, const int c_r4,
                                       std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    return DataFormatConverter::ConvertFromNCHWToNHWC4Int8(reinterpret_cast<int8_t*>(image.GetData()),
                                                           reinterpret_cast<int8_t*>(handle_ptr), batch, channel, hw);
}

// convert from mat to blob
REGISTER_ARM_BLOB_CONVERT_FUNC(N8UC4,               DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertN8UC4ToInt8Blob)
REGISTER_ARM_BLOB_CONVERT_FUNC(N8UC4,               DATA_TYPE_FLOAT, CVT_DIR_MAT2BLOB, ConvertN8UC4ToFloatBlob)
REGISTER_ARM_BLOB_CONVERT_FUNC(N8UC3,               DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertN8UC3ToInt8Blob)
REGISTER_ARM_BLOB_CONVERT_FUNC(N8UC3,               DATA_TYPE_FLOAT, CVT_DIR_MAT2BLOB, ConvertN8UC3ToFloatBlob)
REGISTER_ARM_BLOB_CONVERT_FUNC(NGRAY,               DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertNGRAYToInt8Blob)
REGISTER_ARM_BLOB_CONVERT_FUNC(NGRAY,               DATA_TYPE_FLOAT, CVT_DIR_MAT2BLOB, ConvertNGRAYToFloatBlob)
REGISTER_ARM_BLOB_CONVERT_FUNC(NNV12,               DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertNNV12ToInt8Blob)
REGISTER_ARM_BLOB_CONVERT_FUNC(NNV12,               DATA_TYPE_FLOAT, CVT_DIR_MAT2BLOB, ConvertNNV12ToFloatBlob)
REGISTER_ARM_BLOB_CONVERT_FUNC(NNV21,               DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertNNV21ToInt8Blob)
REGISTER_ARM_BLOB_CONVERT_FUNC(NNV21,               DATA_TYPE_FLOAT, CVT_DIR_MAT2BLOB, ConvertNNV21ToFloatBlob)
REGISTER_ARM_BLOB_CONVERT_FUNC(NCHW_FLOAT,          DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertNCHWFloatToInt8Blob)
REGISTER_ARM_BLOB_CONVERT_FUNC(NCHW_FLOAT,          DATA_TYPE_FLOAT, CVT_DIR_MAT2BLOB, (ConvertFloatMatToFloatBlob<float,float>))
REGISTER_ARM_BLOB_CONVERT_FUNC(NCHW_FLOAT,          DATA_TYPE_BFP16, CVT_DIR_MAT2BLOB, (ConvertFloatMatToFloatBlob<float, bfp16_t>))
REGISTER_ARM_BLOB_CONVERT_FUNC(RESERVED_BFP16_TEST, DATA_TYPE_BFP16, CVT_DIR_MAT2BLOB, (ConvertFloatMatToFloatBlob<bfp16_t, bfp16_t>))
REGISTER_ARM_BLOB_CONVERT_FUNC(RESERVED_FP16_TEST,  DATA_TYPE_FLOAT, CVT_DIR_MAT2BLOB, (ConvertFloatMatToFloatBlob<fp16_t,float>))
REGISTER_ARM_BLOB_CONVERT_FUNC(RESERVED_INT8_TEST,  DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertInt8MatToInt8Blob)

#if TNN_ARM82
static Status ConvertN8UC4ToHalfBlob(Mat& image, char* handle_ptr, const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4, std::vector<float>& fused_int8_scale,
                                     std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        BGRAToBlob(reinterpret_cast<uint8_t*>(image.GetData()) + n * 4 * hw,
                   reinterpret_cast<fp16_t*>(handle_ptr) + n * 8 * hw, param.scale.data(), param.bias.data(), hw,
                   param.reverse_channel, channel);
    }
    return TNN_OK;
}

static Status ConvertN8UC3ToHalfBlob(Mat& image, char* handle_ptr, const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4, std::vector<float>& fused_int8_scale,
                                     std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        BGRToBlob(reinterpret_cast<uint8_t*>(image.GetData()) + n * 3 * hw,
                  reinterpret_cast<fp16_t*>(handle_ptr) + n * 8 * hw, param.scale.data(), param.bias.data(), hw,
                  param.reverse_channel);
    }
    return TNN_OK;
}

static Status ConvertNGRAYToHalfBlob(Mat& image, char* handle_ptr, const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4, std::vector<float>& fused_int8_scale,
                                     std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        GrayToBlob(reinterpret_cast<uint8_t*>(image.GetData()) + n * 1 * hw,
                   reinterpret_cast<fp16_t*>(handle_ptr) + n * 8 * hw, param.scale[0], param.bias[0], hw);
    }
    return TNN_OK;
}

static Status ConvertNNV12ToHalfBlob(Mat& image, char* handle_ptr,
                                     const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4,
                                     std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    Mat bgr = GetBGRFromYUV(image, dims, hw, true);
    return ConvertN8UC3ToHalfBlob(bgr, handle_ptr, param, dims, hw, c_r4, fused_int8_scale, fused_int8_bias);
}

static Status ConvertNNV21ToHalfBlob(Mat& image, char* handle_ptr,
                                     const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4,
                                     std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    Mat bgr = GetBGRFromYUV(image, dims, hw, false);
    return ConvertN8UC3ToHalfBlob(bgr, handle_ptr, param, dims, hw, c_r4, fused_int8_scale, fused_int8_bias);
}

template <typename T_mat>
static Status ConvertFloatMatToHalfBlob(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                        const DimsVector& dims, const int hw, const int c_r4,
                                        std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    auto c_r8    = ROUND_UP(c_r4, 8);
    if (NeedDoScaleBias(param)) {
        for (int n = 0; n < batch; n++) {
            NCHWToBlob(reinterpret_cast<T_mat*>(image.GetData()) + n * channel * hw,
                       reinterpret_cast<fp16_t*>(handle_ptr) + n * c_r8 * hw, channel, hw, nullptr);
            ScaleBias(reinterpret_cast<fp16_t*>(handle_ptr) + n * c_r8 * hw, channel, hw, param.scale.data(),
                      param.bias.data());
        }
    } else {
        for (int n = 0; n < batch; n++) {
            NCHWToBlob(reinterpret_cast<T_mat*>(image.GetData()) + n * channel * hw,
                       reinterpret_cast<fp16_t*>(handle_ptr) + n * c_r8 * hw, channel, hw, nullptr);
        }
    }
    return TNN_OK;
}

REGISTER_ARM_BLOB_CONVERT_FUNC(N8UC4,               DATA_TYPE_HALF,  CVT_DIR_MAT2BLOB, ConvertN8UC4ToHalfBlob)
REGISTER_ARM_BLOB_CONVERT_FUNC(N8UC3,               DATA_TYPE_HALF,  CVT_DIR_MAT2BLOB, ConvertN8UC3ToHalfBlob)
REGISTER_ARM_BLOB_CONVERT_FUNC(NGRAY,               DATA_TYPE_HALF,  CVT_DIR_MAT2BLOB, ConvertNGRAYToHalfBlob)
REGISTER_ARM_BLOB_CONVERT_FUNC(NNV12,               DATA_TYPE_HALF,  CVT_DIR_MAT2BLOB, ConvertNNV12ToHalfBlob)
REGISTER_ARM_BLOB_CONVERT_FUNC(NNV21,               DATA_TYPE_HALF,  CVT_DIR_MAT2BLOB, ConvertNNV21ToHalfBlob)
REGISTER_ARM_BLOB_CONVERT_FUNC(NCHW_FLOAT,          DATA_TYPE_HALF,  CVT_DIR_MAT2BLOB, ConvertFloatMatToHalfBlob<float>)
REGISTER_ARM_BLOB_CONVERT_FUNC(RESERVED_FP16_TEST,  DATA_TYPE_HALF,  CVT_DIR_MAT2BLOB, ConvertFloatMatToHalfBlob<fp16_t>)
#endif

static Status ConvertInt8BlobToN8UC4(Mat& image, char* handle_ptr, const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4, std::vector<float>& fused_int8_scale,
                                     std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        BlobToBGRA(reinterpret_cast<int8_t*>(handle_ptr) + n * 4 * hw,
                   reinterpret_cast<uint8_t*>(image.GetData()) + n * 4 * hw, fused_int8_scale.data(),
                   fused_int8_bias.data(), hw, param.reverse_channel, channel);
    }
    return TNN_OK;
}

static Status ConvertFloatBlobToN8UC4(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                      const DimsVector& dims, const int hw, const int c_r4,
                                      std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        BlobToBGRA(reinterpret_cast<float*>(handle_ptr) + n * 4 * hw,
                   reinterpret_cast<uint8_t*>(image.GetData()) + n * 4 * hw, param.scale.data(), param.bias.data(), hw,
                   param.reverse_channel, channel);
    }
    return TNN_OK;
}

static Status ConvertInt8BlobToN8UC3(Mat& image, char* handle_ptr, const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4, std::vector<float>& fused_int8_scale,
                                     std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        BlobToBGR(reinterpret_cast<int8_t*>(handle_ptr) + n * 4 * hw,
                  reinterpret_cast<uint8_t*>(image.GetData()) + n * 3 * hw, fused_int8_scale.data(),
                  fused_int8_bias.data(), hw, param.reverse_channel);
    }
    return TNN_OK;
}

static Status ConvertFloatBlobToN8UC3(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                      const DimsVector& dims, const int hw, const int c_r4,
                                      std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        BlobToBGR(reinterpret_cast<float*>(handle_ptr) + n * 4 * hw,
                  reinterpret_cast<uint8_t*>(image.GetData()) + n * 3 * hw, param.scale.data(), param.bias.data(), hw,
                  param.reverse_channel);
    }
    return TNN_OK;
}

static Status ConvertInt8BlobToNCHWFloat(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                         const DimsVector& dims, const int hw, const int c_r4,
                                         std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        Int8BlobToNCHW(reinterpret_cast<int8_t*>(handle_ptr) + n * c_r4 * hw,
                       reinterpret_cast<float*>(image.GetData()) + n * channel * hw, channel, hw,
                       fused_int8_scale.data(), fused_int8_bias.data());
    }
    return TNN_OK;
}

template <typename T_mat, typename T_blob>
static Status ConvertFloatBlobToFloatMat(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                         const DimsVector& dims, const int hw, const int c_r4,
                                         std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    if (NeedDoScaleBias(param)) {
        for (int n = 0; n < batch; n++) {
            RawBuffer scale_biased(c_r4 * hw * sizeof(float));
            ScaleBias(reinterpret_cast<T_blob*>(handle_ptr) + n * c_r4 * hw, channel, hw, param.scale.data(),
                      param.bias.data(), scale_biased.force_to<T_blob*>());
            FloatBlobToNCHW(scale_biased.force_to<T_blob*>(),
                            reinterpret_cast<T_mat*>(image.GetData()) + n * channel * hw, channel, hw);
        }
    } else {
        for (int n = 0; n < batch; n++) {
            FloatBlobToNCHW(reinterpret_cast<T_blob*>(handle_ptr) + n * c_r4 * hw,
                            reinterpret_cast<T_mat*>(image.GetData()) + n * channel * hw, channel, hw);
        }
    }
    return TNN_OK;
}

static Status ConvertInt8BlobToInt8Mat(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                       const DimsVector& dims, const int hw, const int c_r4,
                                       std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    return DataFormatConverter::ConvertFromNHWC4ToNCHWInt8(
        reinterpret_cast<int8_t*>(handle_ptr), reinterpret_cast<int8_t*>(image.GetData()), batch, channel, hw);
}

// convert from blob to mat
REGISTER_ARM_BLOB_CONVERT_FUNC(N8UC4,               DATA_TYPE_INT8,  CVT_DIR_BLOB2MAT, ConvertInt8BlobToN8UC4)
REGISTER_ARM_BLOB_CONVERT_FUNC(N8UC4,               DATA_TYPE_FLOAT, CVT_DIR_BLOB2MAT, ConvertFloatBlobToN8UC4)
REGISTER_ARM_BLOB_CONVERT_FUNC(N8UC3,               DATA_TYPE_INT8,  CVT_DIR_BLOB2MAT, ConvertInt8BlobToN8UC3)
REGISTER_ARM_BLOB_CONVERT_FUNC(N8UC3,               DATA_TYPE_FLOAT, CVT_DIR_BLOB2MAT, ConvertFloatBlobToN8UC3)
REGISTER_ARM_BLOB_CONVERT_FUNC(NCHW_FLOAT,          DATA_TYPE_INT8,  CVT_DIR_BLOB2MAT, ConvertInt8BlobToNCHWFloat)
REGISTER_ARM_BLOB_CONVERT_FUNC(NCHW_FLOAT,          DATA_TYPE_FLOAT, CVT_DIR_BLOB2MAT, (ConvertFloatBlobToFloatMat<float,float>))
REGISTER_ARM_BLOB_CONVERT_FUNC(NCHW_FLOAT,          DATA_TYPE_BFP16, CVT_DIR_BLOB2MAT, (ConvertFloatBlobToFloatMat<float, bfp16_t>))
REGISTER_ARM_BLOB_CONVERT_FUNC(RESERVED_BFP16_TEST, DATA_TYPE_BFP16, CVT_DIR_BLOB2MAT, (ConvertFloatBlobToFloatMat<bfp16_t, bfp16_t>))
REGISTER_ARM_BLOB_CONVERT_FUNC(RESERVED_INT8_TEST,  DATA_TYPE_INT8,  CVT_DIR_BLOB2MAT, ConvertInt8BlobToInt8Mat)

#if TNN_ARM82
static Status ConvertHalfBlobToN8UC4(Mat& image, char* handle_ptr, const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4, std::vector<float>& fused_int8_scale,
                                     std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        BlobToBGRA(reinterpret_cast<fp16_t*>(handle_ptr) + n * 8 * hw,
                   reinterpret_cast<uint8_t*>(image.GetData()) + n * 4 * hw, param.scale.data(), param.bias.data(), hw,
                   param.reverse_channel, channel);
    }
    return TNN_OK;
}

static Status ConvertHalfBlobToN8UC3(Mat& image, char* handle_ptr, const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4, std::vector<float>& fused_int8_scale,
                                     std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    for (int n = 0; n < batch; n++) {
        BlobToBGR(reinterpret_cast<fp16_t*>(handle_ptr) + n * 8 * hw,
                  reinterpret_cast<uint8_t*>(image.GetData()) + n * 3 * hw, param.scale.data(), param.bias.data(), hw,
                  param.reverse_channel);
    }
    return TNN_OK;
}

template <typename T_mat>
static Status ConvertHalfBlobToFloatMat(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                        const DimsVector& dims, const int hw, const int c_r4,
                                        std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    auto c_r8    = ROUND_UP(c_r4, 8);
    if (NeedDoScaleBias(param)) {
        for (int n = 0; n < batch; n++) {
            RawBuffer scale_biased(c_r8 * hw * sizeof(fp16_t));
            ScaleBias(reinterpret_cast<fp16_t*>(handle_ptr) + n * c_r8 * hw, channel, hw, param.scale.data(),
                      param.bias.data(), scale_biased.force_to<fp16_t*>());
            HalfBlobToNCHW(scale_biased.force_to<fp16_t*>(),
                           reinterpret_cast<T_mat*>(image.GetData()) + n * channel * hw, channel, hw);
        }
    } else {
        for (int n = 0; n < batch; n++) {
            HalfBlobToNCHW(reinterpret_cast<fp16_t*>(handle_ptr) + n * c_r8 * hw,
                           reinterpret_cast<T_mat*>(image.GetData()) + n * channel * hw, channel, hw);
        }
    }
    return TNN_OK;
}

REGISTER_ARM_BLOB_CONVERT_FUNC(N8UC4,               DATA_TYPE_HALF,  CVT_DIR_BLOB2MAT, ConvertHalfBlobToN8UC4)
REGISTER_ARM_BLOB_CONVERT_FUNC(N8UC3,               DATA_TYPE_HALF,  CVT_DIR_BLOB2MAT, ConvertHalfBlobToN8UC3)
REGISTER_ARM_BLOB_CONVERT_FUNC(NCHW_FLOAT,          DATA_TYPE_HALF,  CVT_DIR_BLOB2MAT, ConvertHalfBlobToFloatMat<float>)
REGISTER_ARM_BLOB_CONVERT_FUNC(RESERVED_FP16_TEST,  DATA_TYPE_HALF,  CVT_DIR_BLOB2MAT, ConvertHalfBlobToFloatMat<fp16_t>)
#endif

}  // namespace TNN_NS
