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

#include <cmath>
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC_WITH_FUNC(LayerNorm, LAYER_LAYER_NORM,
                          virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
                                                                 const std::vector<Blob *> &outputs););

Status CpuLayerNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuLayerNormLayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
                                                     const std::vector<Blob *> &outputs) {
    auto *layer_param = dynamic_cast<LayerNormLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    if (inputs.size() < 3) {
        return Status(TNNERR_PARAM_ERR, "LayerNormLayer has no input blob of scale or bias");
    }

    auto input_blob = inputs[0];
    auto scale_blob = inputs[1];
    auto bias_blob  = inputs[2];
    auto dims_input = input_blob->GetBlobDesc().dims;
    auto dims_scale = scale_blob->GetBlobDesc().dims;
    auto dims_bias  = bias_blob->GetBlobDesc().dims;

    if (!DimsVectorUtils::Equal(dims_scale, dims_bias)) {
        return Status(TNNERR_PARAM_ERR, "LayerNormLayer has invalid dims for input blob of scale or bias");
    }

    // enure dims are valid
    const int dim_offset = (int)dims_input.size() - (int)dims_scale.size();
    for (int i = 0; i < dims_scale.size(); i++) {
        if (dim_offset < 0 || dims_input[i + dim_offset] != dims_scale[i] || dims_scale[i] != dims_bias[i]) {
            return Status(TNNERR_PARAM_ERR, "LayerNormLayer has invalid dims for input blob");
        }
    }

    layer_param->reduce_dims_size = dims_scale.size();

    return TNN_OK;
}

Status CpuLayerNormLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<LayerNormLayerParam *>(param_);
    //Unlike Batch Normalization and Instance Normalization, which applies scalar scale and bias for each entire channel/plane with the affine option,
    //Layer Normalization applies per-element scale and bias with elementwise_affine.
    //see https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html?highlight=layernorm#torch.nn.LayerNorm

    auto input_blob  = inputs[0];
    auto scale_blob  = inputs[1];
    auto bias_blob  = inputs[2];
    auto output_blob = outputs[0];
    auto dims_input = input_blob->GetBlobDesc().dims;

    const int reduce_dim_size = layer_param->reduce_dims_size;

    if (layer_param->reduce_dims_size != scale_blob->GetBlobDesc().dims.size()) {
        return Status(TNNERR_PARAM_ERR, "LayerNormLayer has invalid dims for input blob of scale or bias");
    }

    const int channel_dim_size = (int)dims_input.size() - reduce_dim_size;

    const int channels = DimsVectorUtils::Count(dims_input, 0, channel_dim_size);
    const int channel_area = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, channel_dim_size);
    
    if (0 == channels || 0 == channel_area) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }
    
    float *k_data = (float *)((char*)scale_blob->GetHandle().base + scale_blob->GetHandle().bytes_offset);
    float *b_data = (float *)((char*)bias_blob->GetHandle().base + bias_blob->GetHandle().bytes_offset);

    const float epsilon = layer_param->eps;

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = (float *)((char *)input_blob->GetHandle().base+ input_blob->GetHandle().bytes_offset);
        float *output_data = (float *)((char *)output_blob->GetHandle().base + output_blob->GetHandle().bytes_offset);
        //浮点运算在累加时存在大数吃小数情况，造成误差大，instancenorm累加次数大，更容易出现
        //可考虑用Kahan公式或者用double运算，最后转换成float
        // https://blog.csdn.net/weixin_34268753/article/details/85917630

        //利用方差计算公式减少读次数
        // https://baike.baidu.com/item/方差计算公式/5318566?fr=aladdin
        for (int c = 0; c < channels; c++) {
            //sum_x sum_x2
            double mean_x = 0;
            double variance = 1;
            {
                double sum_x  = 0;
                double sum_x2 = 0;
                for (int hw = 0; hw < channel_area; ++hw) {
                    auto temp = input_data[hw];
                    sum_x += temp;
                    sum_x2 += temp * temp;
                    ;
                }
                mean_x  = sum_x / channel_area;
                auto mean_x2 = sum_x2 / channel_area;

                variance = mean_x2 - mean_x * mean_x;
                variance = 1.0f / sqrt(variance + epsilon);
            }

            for (int hw = 0; hw < channel_area; ++hw, ++output_data, ++input_data) {
                float k = k_data[hw];
                float bias = b_data[hw];
                bias -= mean_x * variance * k;
                
                *output_data = (float)((*input_data) * variance * k + bias);
                    
            }
        }
    } else {
        LOGE("Error: CpuLayerNormLayerAcc layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuLayerNormLayerAcc layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(LayerNorm, LAYER_LAYER_NORM);

}  // namespace TNN_NS
