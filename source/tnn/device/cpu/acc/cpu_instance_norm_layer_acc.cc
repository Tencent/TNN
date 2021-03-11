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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC_WITH_FP32_RESOURCE(InstanceNorm, LAYER_INST_BATCH_NORM);

Status CpuInstanceNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuInstanceNormLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_res = dynamic_cast<InstanceNormLayerResource *>(resource_);
    if (!layer_res) {
        LOGE("Error: layer resource is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer resource is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];

    int batch    = output_blob->GetBlobDesc().dims[0];
    int channels = output_blob->GetBlobDesc().dims[1];
    int area     = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);
    if (0 == area) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    RawBuffer scale_handle = layer_res->scale_handle;
    float *k_data          = layer_res->scale_handle.force_to<float *>();
    float *b_data          = layer_res->bias_handle.force_to<float *>();

    float epsilon = 0.00001f;

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);
        //浮点运算在累加时存在大数吃小数情况，造成误差大，instancenorm累加次数大，更容易出现
        //可考虑用Kahan公式或者用double运算，最后转换成float
        // https://blog.csdn.net/weixin_34268753/article/details/85917630

        //利用方差计算公式减少读次数
        // https://baike.baidu.com/item/方差计算公式/5318566?fr=aladdin
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; ++c) {
                double sum_x  = 0;
                double sum_x2 = 0;
                for (int hw = 0; hw < area; ++hw) {
                    auto temp = input_data[hw];
                    sum_x += temp;
                    sum_x2 += temp * temp;
                    ;
                }
                auto mean_x  = sum_x / area;
                auto mean_x2 = sum_x2 / area;

                auto variance = mean_x2 - mean_x * mean_x;
                variance      = variance > 0 ? variance : 0;
                variance      = 1.0f / sqrt(variance + epsilon);

                double k = k_data[c];
                variance *= k;
                double b = b_data == NULL ? 0.0f : b_data[c];
                b -= mean_x * variance;
                for (int hw = 0; hw < area; ++hw, ++output_data, ++input_data) {
                    *output_data = (float)((*input_data) * variance + b);
                }
            }
        }
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(InstanceNorm, LAYER_INST_BATCH_NORM);

}  // namespace TNN_NS
