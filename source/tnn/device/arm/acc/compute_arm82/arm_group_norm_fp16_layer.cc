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
#include "tnn/device/arm/acc/arm_group_norm_layer_acc.h"
#include "tnn/utils/half.hpp"

namespace TNN_NS {
Status ArmGroupNormLayerAcc::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
    auto layer_param    = dynamic_cast<GroupNormLayerParam *>(param_);
    const float epsilon = layer_param->eps;
    Blob *input_blob    = inputs[0];
    Blob *scale_blob    = inputs[1];
    Blob *bias_blob     = inputs[2];
    Blob *output_blob   = outputs[0];

    const int group              = layer_param->group;
    const int batch_time_group   = output_blob->GetBlobDesc().dims[0] * group;
    const int channels_per_group = output_blob->GetBlobDesc().dims[1] / group;
    const int channel_area       = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);
    const int group_area         = channel_area * channels_per_group;
    if (0 == group_area || 0 == channels_per_group) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    auto k_data = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(scale_blob->GetHandle()));
    auto b_data = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(bias_blob->GetHandle()));

    auto dst = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output_blob->GetHandle()));
    auto src = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input_blob->GetHandle()));

    for (int b = 0; b < batch_time_group; b += 1) {
        Float4 sum_x_f4(0.f);
        Float4 sum_x2_f4(0.f);

        float sum_x  = 0.f;
        float sum_x2 = 0.f;

        // kahan acc, improve accumulation accuracy
        // https://blog.csdn.net/weixin_34268753/article/details/85917630
        Float4 c_x_f4(0.f);
        Float4 c_x2_f4(0.f);

        for (int hw = 0; hw < group_area >> 2 << 2; hw += 4) {
            auto x = Half4::half4_to_float4(Half4::load(src + b * group_area + hw));

            auto y   = x - c_x_f4;
            auto tmp = sum_x_f4 + y;
            c_x_f4   = (tmp - sum_x_f4) - y;
            sum_x_f4 = tmp;

            y         = (x * x) - c_x2_f4;
            tmp       = sum_x2_f4 + y;
            c_x2_f4   = (tmp - sum_x2_f4) - y;
            sum_x2_f4 = tmp;
        }

        for (int hw = group_area >> 2 << 2; hw < group_area; hw += 1) {
            auto x = (float)(src[b * group_area + hw]);
            sum_x  = sum_x + x;
            sum_x2 = sum_x2 + (x * x);
        }

        sum_x  = sum_x + sum_x_f4[0] + sum_x_f4[1] + sum_x_f4[2] + sum_x_f4[3];
        sum_x2 = sum_x2 + sum_x2_f4[0] + sum_x2_f4[1] + sum_x2_f4[2] + sum_x2_f4[3];

        float mean_x   = sum_x / group_area;
        float mean_x2  = sum_x2 / group_area;
        float variance = mean_x2 - mean_x * mean_x;
        variance       = 1 / (sqrt(variance + epsilon));

        int output_channel = (b % group) * channels_per_group;
        for (int c = 0; c < channels_per_group; ++c, ++output_channel) {
            float k    = (float)(k_data[output_channel]);
            float bias = b_data == NULL ? 0.0f : (float)(b_data[output_channel]);
            bias -= mean_x * variance * k;
            auto offset = b * group_area + c * channel_area;
            for (int hw = 0; hw < channel_area >> 2 << 2; hw += 4) {
                auto x   = Half4::half4_to_float4(Half4::load(src + offset + hw));
                auto tmp = x * variance * k + bias;
                Half4::save(dst + offset + hw, Half4::float4_to_half4(tmp));
            }
            for (int hw = channel_area >> 2 << 2; hw < channel_area; hw += 1) {
                auto x           = src[offset + hw];
                auto tmp         = x * variance * k + bias;
                dst[offset + hw] = (fp16_t)tmp;
            }
        }
    }
    return TNN_OK;
}

}  // namespace TNN_NS

#endif  // namespace TNN_ARM82
