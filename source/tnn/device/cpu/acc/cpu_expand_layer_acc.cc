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

#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(Expand, LAYER_EXPAND);

Status CpuExpandLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuExpandLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];
    auto output_dims = output_blob->GetBlobDesc().dims;
    auto input_dims = input_blob->GetBlobDesc().dims;
    int diff = output_dims.size() - input_dims.size();
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);
        int output_diff_start_cnt = DimsVectorUtils::Count(output_dims, diff);
        for(int i = 0; i < output_diff_start_cnt; ++i) {
            int index = i, in_index = 0;
            for(int j = input_dims.size() - 1; j >= 0; ++j) {
                int input_dim = input_dims[j];
                int output_dim = output_dims[j + diff];
                int mod = index % output_dim;
                if(input_dim == 1) {
                    mod = 0;
                }
                index /= output_dim;
                in_index += mod * DimsVectorUtils::Count(input_dims, i);
            }
            output_data[i] = input_data[in_index];
        }
        if(diff > 0) {
            int repeat_cnt = DimsVectorUtils::Count(output_dims, 0, diff);
            for(int i = 1; i < repeat_cnt; ++i) {
                memcpy(output_data + i * output_diff_start_cnt, output_data, output_diff_start_cnt * sizeof(float));
            }
        }
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: expand layer acc dont support datatype");
    }
    
    return TNN_OK;
}

REGISTER_CPU_ACC(Expand, LAYER_EXPAND);

}  // namespace TNN_NS
