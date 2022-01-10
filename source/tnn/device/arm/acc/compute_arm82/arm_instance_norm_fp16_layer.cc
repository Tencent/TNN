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
    if (scale_handle.GetDataType() == DATA_TYPE_FLOAT) {
        scale_handle = ConvertFloatToFP16(scale_handle);
    }
    RawBuffer bias_handle = layer_res->bias_handle;
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
    fp16_t *input_data  = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    fp16_t *output_data = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
    Float4 area_v       = Float4(area);
    Float4 epsilon      = Float4(layer_param->eps);
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < c_r8; c += 8) {
            Float4 sum_low   = Float4(0.0f);
            Float4 sum_high  = Float4(0.0f);
            fp16_t *input_c  = input_data + b * c_r8 * area + c * area;
            fp16_t *output_c = output_data + b * c_r8 * area + c * area;
            for (int hw = 0; hw < area; ++hw) {
                Half8 v      = Half8::load(input_c + hw * 8);
                Half4 v_low  = Half8::get_low(v);
                Half4 v_high = Half8::get_high(v);
                Half4::add_to_f32(v_low, sum_low);
                Half4::add_to_f32(v_high, sum_high);
            }
            Float4 mean_low     = Float4::div(sum_low, area_v);
            Float4 mean_high    = Float4::div(sum_high, area_v);
            Float4 sum_var_low  = Float4(0.0f);
            Float4 sum_var_high = Float4(0.0f);
            for (int hw = 0; hw < area; ++hw) {
                Half8 v          = Half8::load(input_c + hw * 8);
                Float4 diff_low  = Half4::half4_to_float4(Half8::get_low(v)) - mean_low;
                Float4 diff_high = Half4::half4_to_float4(Half8::get_high(v)) - mean_high;
                Float4::mla(sum_var_low, diff_low, diff_low);
                Float4::mla(sum_var_high, diff_high, diff_high);
            }
            Float4 variance_low  = Float4::div(sum_var_low, area_v);
            Float4 variance_high = Float4::div(sum_var_high, area_v);

            variance_low  = Float4::sqrt(variance_low + epsilon);
            variance_high = Float4::sqrt(variance_high + epsilon);
            Half8 k       = Half8::load(k_data + c);
            Float4 k_low  = Half4::half4_to_float4(Half8::get_low(k));
            Float4 k_high = Half4::half4_to_float4(Half8::get_high(k));
            variance_low  = Float4::div(k_low, variance_low);
            variance_high = Float4::div(k_high, variance_high);

            Half8 b       = b_data == nullptr ? Half8((fp16_t)0.0f) : Half8::load(b_data);
            Float4 b_low  = Half4::half4_to_float4(Half8::get_low(b));
            Float4 b_high = Half4::half4_to_float4(Half8::get_high(b));
            Float4::mls(b_low, mean_low, variance_low);
            Float4::mls(b_high, mean_high, variance_high);

            for (int hw = 0; hw < area; ++hw) {
                Half8 v         = Half8::load(input_c + hw * 8);
                Float4 v_low    = Half4::half4_to_float4(Half8::get_low(v));
                Float4 v_high   = Half4::half4_to_float4(Half8::get_high(v));
                Float4 res_low  = b_low;
                Float4 res_high = b_high;
                Float4::mla(res_low, v_low, variance_low);
                Float4::mla(res_high, v_high, variance_high);
                Half8 res = Half8::combine(Half4::float4_to_half4(res_low), Half4::float4_to_half4(res_high));
                Half8::save(output_c + hw * 8, res);
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
