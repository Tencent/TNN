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

#include <math.h>
#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

#include "tnn/device/x86/acc/Float4.h"
#include "tnn/device/x86/acc/Float8.h"
namespace TNN_NS {

DECLARE_X86_ACC(LayerNorm, LAYER_LAYER_NORM);

template <typename VEC, int pack>
static void norm_func(float *input, float *output, int channels, int area, const float *k_data, const float *b_data,
                      float ep) {
    float *input_data  = input;
    float *output_data = output;
    for (int c = 0; c < channels; c++) {
        // step 1: calc varience
        VEC v_sum_x  = VEC(0.f);
        VEC v_sum_x2 = VEC(0.f);
        VEC v_temp   = VEC(0.f);
        float buffer[pack];
        int head       = 0;
        const int tail = area - area % pack;
        double var;

        for (size_t i = head; i < tail; i += pack) {
            v_temp  = VEC::loadu(input_data + i);
            v_sum_x = VEC::add(v_sum_x, v_temp);
            VEC::mla(v_sum_x2, v_temp, v_temp);
        }

        float sum_x = 0.f, sum_x2 = 0.f;
        VEC::saveu(buffer, v_sum_x);
        for (int i = 0; i < pack; i++)
            sum_x += buffer[i];
        VEC::saveu(buffer, v_sum_x2);
        for (int i = 0; i < pack; i++)
            sum_x2 += buffer[i];
        for (size_t i = tail; i < area; i++) {
            var = input_data[i];
            sum_x += var;
            sum_x2 += var * var;
        }

        auto mean_x    = sum_x / area;
        auto mean_x2   = sum_x2 / area;
        float variance = mean_x2 - mean_x * mean_x;
        variance       = variance > 0 ? variance : 0;
        variance       = 1.0f / sqrt(variance + ep);

        // step2: normlization
        VEC v_variance = VEC(&variance);
        VEC v_mean     = VEC(&mean_x);
        for (size_t i = head; i < tail; i += pack, input_data += pack, output_data += pack) {
            VEC v_scale = VEC::loadu(k_data + i);
            VEC v_bias  = VEC::loadu(b_data + i);
            VEC v_data  = VEC::loadu(input_data);

            v_scale = VEC::mul(v_variance, v_scale);  // var * k
            v_bias  = VEC::sub(v_bias, VEC::mul(v_scale, v_mean));

            VEC::mla(v_bias, v_data, v_scale);
            VEC::saveu(output_data, v_bias);
        }

        for (size_t i = tail; i < area; i++, output_data++, input_data++) {
            float b      = b_data[i] - variance * mean_x * k_data[i];
            *output_data = (*input_data) * variance * k_data[i] + b;
        }
    }
}

Status X86LayerNormLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<LayerNormLayerParam *>(param_);
    auto input_blob  = inputs[0];
    auto scale_blob  = inputs[1];
    auto bias_blob   = inputs[2];
    auto output_blob = outputs[0];
    auto dims_input  = input_blob->GetBlobDesc().dims;

    const int reduce_dim_size  = layer_param->reduce_dims_size;
    const int channel_dim_size = (int)dims_input.size() - reduce_dim_size;

    const int channels = DimsVectorUtils::Count(dims_input, 0, channel_dim_size);
    const int area     = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, channel_dim_size);

    if (0 == channels || 0 == area) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    float *k_data = (float *)((char *)scale_blob->GetHandle().base + scale_blob->GetHandle().bytes_offset);
    float *b_data = (float *)((char *)bias_blob->GetHandle().base + bias_blob->GetHandle().bytes_offset);

    const float epsilon = layer_param->eps;

    auto func = norm_func<Float8, 8>;
    if (arch_ == sse42) {
        func = norm_func<Float4, 4>;
    }

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = (float *)((char *)input_blob->GetHandle().base + input_blob->GetHandle().bytes_offset);
        float *output_data = (float *)((char *)output_blob->GetHandle().base + output_blob->GetHandle().bytes_offset);
        func(input_data, output_data, channels, area, k_data, b_data, epsilon);
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_X86_ACC(LayerNorm, LAYER_LAYER_NORM);
}  // namespace TNN_NS