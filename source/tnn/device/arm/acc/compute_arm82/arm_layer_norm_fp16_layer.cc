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

#if TNN_ARM82

#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/device/arm/acc/Half8.h"
#include "tnn/device/arm/acc/arm_layer_norm_layer_acc.h"
#include "tnn/utils/half.hpp"

namespace TNN_NS {
Status ArmLayerNormLayerAcc::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

    auto layer_param             = dynamic_cast<LayerNormLayerParam *>(param_);
    const float epsilon          = layer_param->eps;
    Blob *input_blob             = inputs[0];
    Blob *scale_blob             = inputs[1];
    Blob *bias_blob              = inputs[2];
    Blob *output_blob            = outputs[0];
    const DimsVector &dims_input = input_blob->GetBlobDesc().dims;
    const int reduce_dim_size    = layer_param->reduce_dims_size;
    const int channel_dim_size   = (int)dims_input.size() - reduce_dim_size;
    const int channels           = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims, 0, channel_dim_size);
    const int channel_area       = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, channel_dim_size);
    const int f4_round_down      = channel_area / 4;
    const int f4_remainder       = channel_area % 4;
    if (0 == channels || 0 == channel_area) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    auto src    = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input_blob->GetHandle()));
    auto dst    = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output_blob->GetHandle()));
    auto k_data = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(scale_blob->GetHandle()));
    auto b_data = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(bias_blob->GetHandle()));

    for (int c = 0; c < channels; c += 1) {
        fp16_t *input_ptr  = src + c * channel_area;
        fp16_t *output_ptr = dst + c * channel_area;
        Float4 sum_v       = Float4(0.0f);
        Float4 sum_var_v   = Float4(0.0f);
        Float4 c_x         = Float4(0.0f);
        Float4 c_x_var     = Float4(0.0f);
        for (int hw = 0; hw < f4_round_down * 4; hw += 4) {
            Float4 v = Half4::half4_to_float4(Half4::load(input_ptr + hw));

            Float4 x  = v - c_x;
            Float4 t0 = sum_v + x;
            c_x       = (t0 - sum_v) - x;
            sum_v     = t0;

            Float4 y  = (v * v) - c_x_var;
            Float4 t1 = sum_var_v + y;
            c_x_var   = (t1 - sum_var_v) - y;
            sum_var_v = t1;
        }
        float sum     = 0.0f;
        float sum_var = 0.0f;
        for (int hw = f4_round_down * 4; hw < f4_round_down * 4 + f4_remainder; hw += 1) {
            float v = (float)input_ptr[hw];
            sum     = sum + v;
            sum_var = sum_var + v * v;
        }
        sum     = sum + sum_v[0] + sum_v[1] + sum_v[2] + sum_v[3];
        sum_var = sum_var + sum_var_v[0] + sum_var_v[1] + sum_var_v[2] + sum_var_v[3];

        float mean     = sum / channel_area;
        float mean_var = sum_var / channel_area;
        float variance = mean_var - mean * mean;
        variance       = 1 / std::sqrt(variance + epsilon);

        for (int hw = 0; hw < f4_round_down * 4; hw += 4) {
            Float4 k    = Half4::half4_to_float4(Half4::load(k_data + hw));
            Float4 bias = Half4::half4_to_float4(Half4::load(b_data + hw));
            Float4::mls(bias, variance * mean, k);

            Float4 x   = Half4::half4_to_float4(Half4::load(input_ptr + hw));
            Float4 res = ((x * variance) * k) + bias;
            Half4::save(output_ptr + hw, Half4::float4_to_half4(res));
        }
        for (int hw = f4_round_down * 4; hw < f4_round_down * 4 + f4_remainder; hw += 1) {
            auto k         = (float)k_data[hw];
            auto bias      = (float)b_data[hw];
            bias           = bias - ((mean * variance) * k);
            auto x         = (float)input_ptr[hw];
            float res      = ((x * variance) * k) + bias;
            output_ptr[hw] = (fp16_t)res;
        }
    }
    return TNN_OK;
}

}  // namespace TNN_NS

#endif  // namespace TNN_ARM82
