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

#ifdef TNN_ARM82

#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/device/arm/acc/Half8.h"
#include "tnn/device/arm/acc/arm_instance_norm_layer_acc.h"
#include "tnn/utils/half.hpp"
#include <arm_neon.h>
#include <arm_fp16.h>

namespace TNN_NS {
Status ArmInstanceNormLayerAcc::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<InstanceNormLayerParam *>(param_);
    auto layer_res   = dynamic_cast<InstanceNormLayerResource *>(resource_);
    if (!layer_res || !layer_param) {
        return Status(TNNERR_MODEL_ERR, "Error: layer param or resource is nil");
    }
    auto desc    = outputs[0]->GetBlobDesc();
    int batch    = desc.dims[0];
    int channels = desc.dims[1];
    int c_r8     = ROUND_UP(channels, 8);
    int area     = DimsVectorUtils::Count(desc.dims, 2);

    RawBuffer scale_handle = layer_res->scale_handle;
    RawBuffer bias_handle  = layer_res->bias_handle;
    if (scale_handle.GetDataType() == DATA_TYPE_FLOAT) {
        scale_handle = ConvertFloatToFP16(scale_handle);
    }
    if (bias_handle.GetDataType() == DATA_TYPE_FLOAT) {
        bias_handle = ConvertFloatToFP16(bias_handle);
    }
    auto *k_data = scale_handle.force_to<fp16_t *>();
    auto *b_data = bias_handle.force_to<fp16_t *>();

    if (channels != c_r8) {
        fp16_t *tmp = new fp16_t[c_r8];
        memcpy(tmp, k_data, channels * sizeof(fp16_t));
        memset(tmp + channels, 0, (c_r8 - channels) * sizeof(fp16_t));
        k_data = tmp;
        if (b_data) {
            tmp = new fp16_t[c_r8];
            memcpy(tmp, b_data, channels * sizeof(fp16_t));
            memset(tmp + channels, 0, (c_r8 - channels) * sizeof(fp16_t));
            b_data = tmp;
        }
    }

    float16_t *input_data  = reinterpret_cast<float16_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    float16_t *output_data = reinterpret_cast<float16_t *>(GetBlobHandlePtr(outputs[0]->GetHandle()));


    float32x4_t area_v  = vdupq_n_f32(area);
    float32x4_t epsilon = vdupq_n_f32(0.00001f);
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < c_r8; c += 8) {
            float32x4_t sum_low  = vdupq_n_f32(0.0f);
            float32x4_t sum_high = vdupq_n_f32(0.0f);
            fp16_t *input_c      = input_data + b * c_r8 * area + c * area;
            fp16_t *output_c     = output_data + b * c_r8 * area + c * area;
            for (int hw = 0; hw < area; ++hw) {
                float16x8_t v = vld1q_f16(input_c + hw * 8);
                float16x4_t v_low = vld1_f16(input_c + hw *8);

                sum_low       = sum_low + vcvt_f32_f16(vget_low_f16(v));
                sum_high      = sum_high + vcvt_f32_f16(vget_high_f16(v));
            }
            float32x4_t mean_low  = vdivq_f32(sum_low, area_v);
            float32x4_t mean_high = vdivq_f32(sum_high, area_v);

            float32x4_t sum_var_low  = vdupq_n_f32(0.0f);
            float32x4_t sum_var_high = vdupq_n_f32(0.0f);
            for (int hw = 0; hw < area; ++hw) {
                float16x8_t v         = vld1q_f16(input_c + hw * 8);
                float32x4_t diff_low  = vcvt_f32_f16(vget_low_f16(v)) - mean_low;
                float32x4_t diff_high = vcvt_f32_f16(vget_high_f16(v)) - mean_high;
                sum_var_low           = sum_var_low + vmulq_f32(diff_low, diff_low);
                sum_var_high          = sum_var_high + vmulq_f32(diff_high, diff_high);
            }
            float32x4_t variance_low  = vdivq_f32(sum_var_low, area_v);
            float32x4_t variance_high = vdivq_f32(sum_var_high, area_v);

            variance_low  = vsqrtq_f32(variance_low + epsilon);
            variance_high = vsqrtq_f32(variance_high + epsilon);
            float16x8_t k = vld1q_f16(k_data + c);

            variance_low       = vdivq_f32(vcvt_f32_f16(vget_low_f16(k)), variance_low);
            variance_high      = vdivq_f32(vcvt_f32_f16(vget_high_f16(k)), variance_high);
            float16x8_t b      = b_data == nullptr ? vdupq_n_f32(0.0f) : vld1q_f16(b_data);
            float32x4_t b_low  = vcvt_f32_f16(vget_low_f16(b)) - vmulq_f32(mean_low, variance_low);
            float32x4_t b_high = vcvt_f32_f16(vget_high_f16(b)) - vmulq_f32(mean_high, variance_high);
            for (int hw = 0; hw < area; ++hw) {
                float16x8_t v        = vld1q_f16(input_c + hw * 8);
                float32x4_t res_low  = vcvt_f32_f16(vget_low_f16(v)) * variance_low + b_low;
                float32x4_t res_high = vcvt_f32_f16(vget_high_f16(v)) * variance_high + b_high;
                float16x8_t res      = vcombine_f16(vcvt_f16_f32(res_low), vcvt_f16_f32(res_high));
                vst1q_f16(output_c + hw * 8, res);
            }
        }
    }
    if (channels != c_r8) {
        delete[] k_data;
        if (b_data) {
            delete[] b_data;
        }
    }
    return TNN_OK;
}

}  // namespace TNN_NS

#endif  // namespace TNN_ARM82
