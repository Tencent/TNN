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

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(PadV2, LAYER_PADV2);

Status CpuPadV2LayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

void ConstPadV2(float* input_data, float* output_data, DimsVector input_dims, DimsVector output_dims,
              PadLayerParam* layer_param) {
    float value = layer_param->value;
    
    const int count = DimsVectorUtils::Count(output_dims);
    DimsVector output_dim_index(output_dims.size(), 0);
    for (int i = 0; i<count; i++) {
        auto input_index = DimsVectorUtils::Pad(output_dim_index, input_dims, layer_param->pads, layer_param->type, nullptr);
        if (DimsVectorUtils::IsInBox(input_index, input_dims)) {
            int input_offset = DimsOffsetUtils::ConvertIndexToOffset(input_dims, input_index);
            output_data[i] = input_data[input_offset];
        } else {
            output_data[i] = value;
        }
        
        output_dim_index = DimsVectorUtils::IncreaseIndex(output_dim_index, output_dims);
    }
}

Status CpuPadV2LayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PadLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: layer param is nil");
    }
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];

    auto input_dims  = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;
    
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT ||
        output_blob->GetBlobDesc().data_type == DATA_TYPE_INT32 ||
        output_blob->GetBlobDesc().data_type == DATA_TYPE_UINT32) {
        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);

        if (layer_param->type == 0) {
            // mode: const
            ConstPadV2(input_data, output_data, input_dims, output_dims, layer_param);
        } else {
            LOGE("Error: layer param is not supported: type:%d\n", layer_param->type);
            return Status(TNNERR_PARAM_ERR, "Error: layer param is not supported");
        }
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(PadV2, LAYER_PADV2);

}  // namespace TNN_NS
