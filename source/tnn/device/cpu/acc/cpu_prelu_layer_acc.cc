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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC_WITH_FP32_RESOURCE(PRelu, LAYER_PRELU);

Status CpuPReluLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuPReluLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PReluLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: PReluLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: PReluLayerParam is nil");
    }

    auto layer_res = dynamic_cast<PReluLayerResource *>(resource_);
    if (!layer_res) {
        LOGE("Error: PReluLayerResource is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: PReluLayerResource is nil");
    }

    const int slope_size     = layer_res->slope_handle.GetBytesSize();
    const DataType data_type = layer_res->slope_handle.GetDataType();

    Blob *input_blob       = inputs[0];
    Blob *output_blob      = outputs[0];
    int channel            = output_blob->GetBlobDesc().dims[1];
    int count              = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    const int channel_size = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);
    if (0 == channel_size) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    if (output_blob->GetBlobDesc().data_type != DATA_TYPE_INT8) {
        const float *slope_data = layer_res->slope_handle.force_to<float *>();

        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);
        if (layer_param->channel_shared) {
            for (int index = 0; index < count; ++index) {
                if (input_data[index] < 0) {
                    output_data[index] = input_data[index] * slope_data[0];
                } else {
                    output_data[index] = input_data[index];
                }
            }
        } else {
            for (int index = 0; index < count; ++index) {
                if (input_data[index] < 0) {
                    int channel_index  = (index / channel_size) % channel;
                    output_data[index] = input_data[index] * slope_data[channel_index];
                } else {
                    output_data[index] = input_data[index];
                }
            }
        }
    } else {
        ASSERT(0);
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(PRelu, LAYER_PRELU);

}  // namespace TNN_NS
