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
#include "tnn/device/arm/acc/arm_layer_norm_layer_acc.h"
#include "tnn/utils/half.hpp"

namespace TNN_NS {
Status ArmLayerNormLayerAcc::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

    auto layer_param = dynamic_cast<LayerNormLayerParam *>(param_);
    // const float epsilon          = layer_param->eps;
    half_float::detail::uint16 epsilon_tmp =
        half_float::detail::float2half<(std::float_round_style)(HALF_ROUND_STYLE)>(layer_param->eps);
    fp16_t epsion                = *(fp16_t *)(&epsilon_tmp);
    Blob *input_blob             = inputs[0];
    Blob *scale_blob             = inputs[1];
    Blob *bias_blob              = inputs[2];
    Blob *output_blob            = outputs[0];
    const DimsVector &dims_input = input_blob->GetBlobDesc().dims;
    const int reduce_dim_size    = layer_param->reduce_dims_size;
    const int channel_dim_size   = (int)dims_input.size() - reduce_dim_size;
    const int channels           = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims, 0, channel_dim_size);
    const int channel_area       = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, channel_dim_size);
    const int f8_round_down      = channel_area / 8;
    const int f8_remainder       = channel_area % 8;
    if (0 == channels || 0 == channel_area) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    auto src    = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input_blob->GetHandle()));
    auto dst    = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output_blob->GetHandle()));
    auto k_data = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(scale_blob->GetHandle()));
    auto b_data = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(bias_blob->GetHandle()));

#if 0
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
#else
    for (int c = 0; c < channels; c += 1) {
        fp16_t *input_ptr  = src + c * channel_area;
        fp16_t *output_ptr = dst + c * channel_area;
        Half8 sum_x_f8     = Half8((fp16_t)0.0f);
        Half8 sum_x2_f8    = Half8((fp16_t)0.0f);
        Half8 c_x_f8       = Half8((fp16_t)0.0f);
        Half8 c_x2_f8      = Half8((fp16_t)0.0f);
        for (int hw = 0; hw < f8_round_down * 8; hw += 8) {
            Half8 v = Half8::load(input_ptr + hw);

            Half8 x  = v - c_x_f8;
            Half8 t0 = sum_x_f8 + x;
            c_x_f8   = (t0 - sum_x_f8) - x;
            sum_x_f8 = t0;

            Half8 y   = (v * v) - c_x2_f8;
            Half8 t1  = sum_x2_f8 + y;
            c_x2_f8   = (t1 - sum_x2_f8) - y;
            sum_x2_f8 = t1;
        }
        fp16_t sum_x  = (fp16_t)0.0f;
        fp16_t sum_x2 = (fp16_t)0.0f;
        for (int hw = f8_round_down * 8; hw < f8_round_down * 8 + f8_remainder; hw += 1) {
            fp16_t v = input_ptr[hw];
            sum_x    = sum_x + v;
            sum_x2   = sum_x2 + v * v;
        }
        for (int i = 0; i < 8; ++i) {
            sum_x  = sum_x + sum_x_f8[i];
            sum_x2 = sum_x2 + sum_x2_f8[i];
        }
        fp16_t mean_x     = fp16_t(sum_x / channel_area);
        fp16_t mean_x2    = fp16_t(sum_x2 / channel_area);
        fp16_t variance   = mean_x2 - mean_x * mean_x;
        variance          = 1 / half_float::detail::sqrt(variance + epsion);
        Half8 mean_x_f8   = Half8(mean_x);
        Half8 variance_f8 = Half8(variance);
        for (int hw = 0; hw < f8_round_down * 8; hw += 8) {
            Half8 k    = Half8::load(k_data + hw);
            Half8 bias = Half8::load(b_data + hw);
            bias       = bias - (mean_x_f8 * variance_f8) * k;

            Half8 x   = Half8::load(input_ptr + hw);
            Half8 res = ((x * variance_f8) * k) + bias;
            Half8::save(output_ptr + hw, res);
        }
        for (int hw = f8_round_down * 8; hw < f8_round_down * 8 + f8_remainder; hw += 1) {
            auto k         = k_data[hw];
            auto bias      = b_data[hw];
            bias           = bias - ((mean_x * variance) * k);
            fp16_t x       = input_ptr[hw];
            fp16_t res     = ((x * variance) * k) + bias;
            output_ptr[hw] = res;
        }
    }
#endif
    return TNN_OK;
}

}  // namespace TNN_NS

#endif  // namespace TNN_ARM82
