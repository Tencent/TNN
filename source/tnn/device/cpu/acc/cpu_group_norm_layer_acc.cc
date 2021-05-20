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

DECLARE_CPU_ACC(GroupNorm, LAYER_GROUP_NORM);

Status CpuGroupNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuGroupNormLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<GroupNormLayerParam *>(param_);
    // for group norm
    // if group == input_channel, it is equivalent with InstanceNorm
    // if group == 1, it is equivalent with LayerNorm

    Blob *input_blob  = inputs[0];
    Blob *scale_blob  = inputs[1];
    Blob *bias_blob  = inputs[2];
    Blob *output_blob = outputs[0];
    
    const int group = layer_param->group;
    const int batch_time_group = output_blob->GetBlobDesc().dims[0] * group;
    const int channels_per_group = output_blob->GetBlobDesc().dims[1] / group;
    const int channel_area = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);
    const int group_area = channel_area * channels_per_group;
    if (0 == group_area || 0 == channels_per_group) {
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
        for (int b = 0; b < batch_time_group; b++) {
            //sum_x sum_x2
            double mean_x = 0;
            double variance = 1;
            {
                double sum_x  = 0;
                double sum_x2 = 0;
                for (int hw = 0; hw < group_area; ++hw) {
                    auto temp = input_data[hw];
                    sum_x += temp;
                    sum_x2 += temp * temp;
                    ;
                }
                mean_x  = sum_x / group_area;
                auto mean_x2 = sum_x2 / group_area;

                variance = mean_x2 - mean_x * mean_x;
                variance = 1.0f / sqrt(variance + epsilon);
            }

            int output_channel = (b % group) * channels_per_group;
            for (int c = 0; c < channels_per_group; ++c, ++output_channel) {
                float k = k_data[output_channel];
                float bias = b_data == NULL ? 0.0f : b_data[output_channel];
                bias -= mean_x * variance * k;
                for (int hw = 0; hw < channel_area; ++hw, ++output_data, ++input_data) {
                    *output_data = (float)((*input_data) * variance * k + bias);
                }
            }
        }
    } else {
        LOGE("Error: CpuGroupNormLayerAcc layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuGroupNormLayerAcc layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(GroupNorm, LAYER_GROUP_NORM);

}  // namespace TNN_NS
