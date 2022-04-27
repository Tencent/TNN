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

#include "arm_layer_norm_layer_acc.h"

#include <cmath>

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

Status ArmLayerNormLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
    auto k_data = reinterpret_cast<float *>(GetBlobHandlePtr(scale_blob->GetHandle()));
    auto b_data = reinterpret_cast<float *>(GetBlobHandlePtr(bias_blob->GetHandle()));

    auto dst = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));
    auto src = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));

    for (int c = 0; c < channels; c += 1) {
        Float4 sum_x_f4(0.f);
        Float4 sum_x2_f4(0.f);

        float sum_x  = 0.f;
        float sum_x2 = 0.f;

        // kahan acc, improve accumulation accuracy
        // https://blog.csdn.net/weixin_34268753/article/details/85917630
        Float4 c_x_f4(0.f);
        Float4 c_x2_f4(0.f);

        for (int hw = 0; hw < f4_round_down * 4; hw += 4) {
            auto x = Float4::load(src + c * channel_area + hw);

            auto y   = x - c_x_f4;
            auto tmp = sum_x_f4 + y;
            c_x_f4   = (tmp - sum_x_f4) - y;
            sum_x_f4 = tmp;

            y         = (x * x) - c_x2_f4;
            tmp       = sum_x2_f4 + y;
            c_x2_f4   = (tmp - sum_x2_f4) - y;
            sum_x2_f4 = tmp;
        }

        for (int hw = f4_round_down * 4; hw < f4_round_down * 4 + f4_remainder; hw += 1) {
            auto x = src[c * channel_area + hw];
            sum_x  = sum_x + x;
            sum_x2 = sum_x2 + (x * x);
        }

        sum_x  = sum_x + sum_x_f4[0] + sum_x_f4[1] + sum_x_f4[2] + sum_x_f4[3];
        sum_x2 = sum_x2 + sum_x2_f4[0] + sum_x2_f4[1] + sum_x2_f4[2] + sum_x2_f4[3];

        float mean_x   = sum_x / channel_area;
        float mean_x2  = sum_x2 / channel_area;
        float variance = mean_x2 - mean_x * mean_x;
        variance       = 1 / (sqrt(variance + epsilon));

        Float4 mean_x_f4(mean_x);
        Float4 variance_f4(variance);

        for (int hw = 0; hw < f4_round_down * 4; hw += 4) {
            auto k    = Float4::load(k_data + hw);
            auto bias = Float4::load(b_data + hw);
            bias      = bias - ((mean_x_f4 * variance_f4) * k);

            auto x   = Float4::load(src + c * channel_area + hw);
            auto tmp = ((x * variance_f4) * k) + bias;
            Float4::save(dst + c * channel_area + hw, tmp);
        }

        for (int hw = f4_round_down * 4; hw < f4_round_down * 4 + f4_remainder; hw += 1) {
            auto k    = k_data[hw];
            auto bias = b_data[hw];
            bias      = bias - ((mean_x * variance) * k);

            auto x   = src[c * channel_area + hw];
            auto tmp = ((x * variance) * k) + bias;

            dst[c * channel_area + hw] = tmp;
        }
    }
    return TNN_OK;
}

Status ArmLayerNormLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    DataType input_data_type = inputs[0]->GetBlobDesc().data_type;
    if (DATA_TYPE_FLOAT == input_data_type) {
        return Exec(inputs, outputs);
    }
#if TNN_ARM82
    else if (DATA_TYPE_HALF == input_data_type) {
        return ExecFp16(inputs, outputs);
    }
#endif
    else {
        LOGE("Error: ArmLayerNormLayerAcc layer acc dont support datatype: %d\n", input_data_type);
        return Status(TNNERR_MODEL_ERR, "Error: ArmLayerNormLayerAcc layer acc dont support datatype");
    }
}
REGISTER_ARM_ACC(LayerNorm, LAYER_LAYER_NORM)
REGISTER_ARM_LAYOUT(LAYER_LAYER_NORM, DATA_FORMAT_NCHW)
REGISTER_ARM_PRECISION_FP16(LAYER_LAYER_NORM)

}  // namespace TNN_NS