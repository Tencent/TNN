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

static inline int upsample_bilinear2d(float *output_data, const float *input_data, int ih, int iw, int oh, int ow,
                                      int c_4, bool align_corners) {
    auto src_z_step = iw * ih * 4;
    auto dst_z_step = ow * oh * 4;
    auto src_y_step = iw * 4;

    RawBuffer h_coeffs(oh * sizeof(float));
    RawBuffer w_coeffs(ow * sizeof(float));
    auto h_coeffs_ptr = h_coeffs.force_to<float *>();
    auto w_coeffs_ptr = w_coeffs.force_to<float *>();

    get_bilinear_coeffs(h_coeffs_ptr, w_coeffs_ptr, ih, iw, oh, ow, align_corners);

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
            const float *Xdata   = &(input_data[h1 * iw * 4 + w1 * 4]);
            float *Ydata         = &(output_data[h2 * ow * 4 + w2 * 4]);
            for (int z = 0; z < c_4; z++) {
                Float4::save(Ydata,
                             (Float4::load(Xdata) * w0lambda + Float4::load(Xdata + w1p * 4) * w1lambda) * h0lambda +
                                 (Float4::load(Xdata + h1p * src_y_step) * w0lambda +
                                  Float4::load(Xdata + h1p * src_y_step + w1p * 4) * w1lambda) *
                                     h1lambda);

                Xdata += src_z_step;
                Ydata += dst_z_step;
            }
        }
    }

    return 0;
}

template <bool do_scale>
static int upsample_bilinear2d(int8_t *output_data, const int8_t *input_data, int ih, int iw, int oh, int ow, int c_4,
                               bool align_corners, const float *scale) {
    auto c_r4       = c_4 * 4;
    auto src_y_step = iw * c_r4;
    auto dst_y_step = ow * c_r4;

    RawBuffer h_coeffs(oh * sizeof(float));
    RawBuffer w_coeffs(ow * sizeof(float));
    auto h_coeffs_ptr = h_coeffs.force_to<float *>();
    auto w_coeffs_ptr = w_coeffs.force_to<float *>();

    get_bilinear_coeffs(h_coeffs_ptr, w_coeffs_ptr, ih, iw, oh, ow, align_corners);

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
                    int8x8_t data00      = vld1_s8(Xdata00);
                    int8x8_t data01      = vld1_s8(Xdata01);
                    int8x8_t data10      = vld1_s8(Xdata10);
                    int8x8_t data11      = vld1_s8(Xdata11);
                    int16x8_t data00h    = vmovl_s8(data00);
                    int16x8_t data01h    = vmovl_s8(data01);
                    int16x8_t data10h    = vmovl_s8(data10);
                    int16x8_t data11h    = vmovl_s8(data11);
                    float32x4_t data00_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(data00h)));
                    float32x4_t data00_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data00h)));
                    float32x4_t data01_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(data01h)));
                    float32x4_t data01_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data01h)));
                    float32x4_t data10_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(data10h)));
                    float32x4_t data10_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data10h)));
                    float32x4_t data11_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(data11h)));
                    float32x4_t data11_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data11h)));
                    float32x4_t acc0     = vmlaq_f32(vmulq_f32(data00_0, v_w0lambda), data01_0, v_w1lambda);
                    float32x4_t acc1     = vmlaq_f32(vmulq_f32(data10_0, v_w0lambda), data11_0, v_w1lambda);
                    float32x4_t acc   = vmulq_f32(vmlaq_f32(vmulq_f32(acc0, v_h0lambda), acc1, v_h1lambda), v_scale0);
                    int16x4_t res_s16 = vqmovn_s32(VCVTAQ_S32_F32(acc));
                    acc0              = vmlaq_f32(vmulq_f32(data00_1, v_w0lambda), data01_1, v_w1lambda);
                    acc1              = vmlaq_f32(vmulq_f32(data10_1, v_w0lambda), data11_1, v_w1lambda);
                    acc               = vmulq_f32(vmlaq_f32(vmulq_f32(acc0, v_h0lambda), acc1, v_h1lambda), v_scale1);
                    vst1_s8(Ydata, vqmovn_s16(VQMOVN_HIGH_S32_T(res_s16, VCVTAQ_S32_F32(acc))));

                    Xdata00 += 8;
                    Xdata01 += 8;
                    Xdata10 += 8;
                    Xdata11 += 8;
                    Ydata += 8;
                    scale_p += 8;
                }
                if (c_4 % 2) {
                    float32x4_t v_scale  = vld1q_f32(scale_p);
                    int8x8_t data00      = vld1_s8(Xdata00);
                    int8x8_t data01      = vld1_s8(Xdata01 - 4);
                    int8x8_t data10      = vld1_s8(Xdata10 - 4);
                    int8x8_t data11      = vld1_s8(Xdata11 - 4);
                    int16x8_t data00h    = vmovl_s8(data00);
                    int16x8_t data01h    = vmovl_s8(data01);
                    int16x8_t data10h    = vmovl_s8(data10);
                    int16x8_t data11h    = vmovl_s8(data11);
                    float32x4_t data00_1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(data00h)));
                    float32x4_t data01_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data01h)));
                    float32x4_t data10_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data10h)));
                    float32x4_t data11_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(data11h)));
                    float32x4_t acc0     = vmlaq_f32(vmulq_f32(data00_1, v_w0lambda), data01_1, v_w1lambda);
                    float32x4_t acc1     = vmlaq_f32(vmulq_f32(data10_1, v_w0lambda), data11_1, v_w1lambda);
                    float32x4_t acc      = vmulq_f32(vmlaq_f32(vmulq_f32(acc0, v_h0lambda), acc1, v_h1lambda), v_scale);
                    int16x4_t res_s16    = vqmovn_s32(VCVTAQ_S32_F32(acc));
                    int8x8_t res_s8      = vqmovn_s16(vcombine_s16(res_s16, res_s16));
                    vst1_lane_s32(reinterpret_cast<int32_t *>(Ydata), vreinterpret_s32_s8(res_s8), 0);
                }
            } else {
                int16x4_t v_w0 = vdup_n_s16(w0_short);
                int16x4_t v_w1 = vdup_n_s16(w1_short);
                int16x4_t v_h0 = vdup_n_s16(h0_short);
                int16x4_t v_h1 = vdup_n_s16(h1_short);
                int32x4_t v_2  = vdupq_n_s32(2);
                for (int z = 0; z < c_4 / 2; z++) {
                    int8x8_t data00   = vld1_s8(Xdata00);
                    int8x8_t data01   = vld1_s8(Xdata01);
                    int8x8_t data10   = vld1_s8(Xdata10);
                    int8x8_t data11   = vld1_s8(Xdata11);
                    int16x8_t data00h = vmovl_s8(data00);
                    int16x8_t data01h = vmovl_s8(data01);
                    int16x8_t data10h = vmovl_s8(data10);
                    int16x8_t data11h = vmovl_s8(data11);
                    int32x4_t acc0    = vmlal_s16(vmull_s16(vget_low_s16(data00h), v_w0), vget_low_s16(data01h), v_w1);
                    int32x4_t acc1    = vmlal_s16(vmull_s16(vget_low_s16(data10h), v_w0), vget_low_s16(data11h), v_w1);
                    int32x4_t acc     = v_2;
                    acc = vsraq_n_s32(v_2, vmlal_s16(vmull_s16(vshrn_n_s32(acc0, 4), v_h0), vshrn_n_s32(acc1, 4), v_h1),
                                      16);
                    int16x4_t res_s16 = vshrn_n_s32(acc, 2);
                    acc0 = vmlal_s16(vmull_s16(vget_high_s16(data00h), v_w0), vget_high_s16(data01h), v_w1);
                    acc1 = vmlal_s16(vmull_s16(vget_high_s16(data10h), v_w0), vget_high_s16(data11h), v_w1);
                    acc  = v_2;
                    acc = vsraq_n_s32(v_2, vmlal_s16(vmull_s16(vshrn_n_s32(acc0, 4), v_h0), vshrn_n_s32(acc1, 4), v_h1),
                                      16);
                    vst1_s8(Ydata, vqmovn_s16(vcombine_s16(res_s16, vshrn_n_s32(acc, 2))));

                    Xdata00 += 8;
                    Xdata01 += 8;
                    Xdata10 += 8;
                    Xdata11 += 8;
                    Ydata += 8;
                }
                if (c_4 % 2) {
                    int8x8_t data00   = vld1_s8(Xdata00);
                    int8x8_t data01   = vld1_s8(Xdata01 - 4);
                    int8x8_t data10   = vld1_s8(Xdata10 - 4);
                    int8x8_t data11   = vld1_s8(Xdata11 - 4);
                    int16x8_t data00h = vmovl_s8(data00);
                    int16x8_t data01h = vmovl_s8(data01);
                    int16x8_t data10h = vmovl_s8(data10);
                    int16x8_t data11h = vmovl_s8(data11);
                    int32x4_t acc0    = vmlal_s16(vmull_s16(vget_low_s16(data00h), v_w0), vget_high_s16(data01h), v_w1);
                    int32x4_t acc1 = vmlal_s16(vmull_s16(vget_high_s16(data10h), v_w0), vget_high_s16(data11h), v_w1);
                    int32x4_t acc  = v_2;
                    acc = vsraq_n_s32(v_2, vmlal_s16(vmull_s16(vshrn_n_s32(acc0, 4), v_h0), vshrn_n_s32(acc1, 4), v_h1),
                                      16);
                    int16x4_t res_s16 = vshrn_n_s32(acc, 2);
                    int8x8_t res_s8   = vqmovn_s16(vcombine_s16(res_s16, res_s16));
                    vst1_lane_s32(reinterpret_cast<int32_t *>(Ydata), vreinterpret_s32_s8(res_s8), 0);
                }
            }
#endif
        }
    }

    return 0;
}

ArmUpsampleLayerAcc::~ArmUpsampleLayerAcc() {}

Status ArmUpsampleLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<UpsampleLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

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
    }

    float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

    auto oc_4 = UP_DIV(dims_output[1], 4);

    if (dims_input[2] == dims_output[2] && dims_input[3] == dims_output[3] && data_type != DATA_TYPE_INT8) {
        if (output_data != input_data) {
            memcpy(output_data, input_data,
                   oc_4 * dims_input[2] * dims_input[3] * 4 * DataTypeUtils::GetBytesSize(data_type));
        }
    } else if (param->mode == 1) {  // nearest
        if (data_type == DATA_TYPE_FLOAT) {
            upsample_nearest2d(output_data, input_data, dims_input[2], dims_input[3], dims_output[2], dims_output[3],
                               oc_4);
        } else {
            return Status(TNNERR_LAYER_ERR, "Error: Not supported data type for upsample nearest");
        }
    } else if (param->mode == 2) {  // bilinear/linear
        if (data_type == DATA_TYPE_FLOAT) {
            upsample_bilinear2d(output_data, input_data, dims_input[2], dims_input[3], dims_output[2], dims_output[3],
                                oc_4, (bool)param->align_corners);
        } else if (data_type == DATA_TYPE_INT8) {
            if (do_scale_)
                upsample_bilinear2d<true>(reinterpret_cast<int8_t *>(output_data),
                                          reinterpret_cast<int8_t *>(input_data), dims_input[2], dims_input[3],
                                          dims_output[2], dims_output[3], oc_4, (bool)param->align_corners,
                                          buffer_scale_.force_to<float *>());
            else
                upsample_bilinear2d<false>(reinterpret_cast<int8_t *>(output_data),
                                           reinterpret_cast<int8_t *>(input_data), dims_input[2], dims_input[3],
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
