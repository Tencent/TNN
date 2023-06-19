// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

DECLARE_CPU_ACC(Roll, LAYER_ROLL);

Status CpuRollLayerAcc::Reshape(const std::vector<Blob*> &inputs, const std::vector<Blob*> &outputs) {
    return TNN_OK;
}

Status CpuRollLayerAcc::Forward(const std::vector<Blob*> &inputs, const std::vector<Blob*> &outputs) {
    // Operator Roll input.dim == output.dim
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];
    auto input_dims  = input_blob->GetBlobDesc().dims;

    auto roll_param  = dynamic_cast<RollLayerParam*>(param_);
    if (roll_param == nullptr) {
        LOGE("Error: CpuRollLayer forward load layer param failed\n");
        return Status(TNNERR_MODEL_ERR, "Error: CpuRollLayer forward Load layer param failed!");
    }
    if (roll_param->dims.size() != roll_param->shifts.size()) {
        LOGE("Error: CpuRollLayer forward layer param.shifts.nbDims not equal to input param.dims.nbDims.\n");
        return Status(TNNERR_MODEL_ERR, "Error: CpuRollLayer forward layer param.shifts.nbDims not equal to input param.dims.nbDims!");
    }
 
    char *input_data   = reinterpret_cast<char *>(input_blob->GetHandle().base);
    char *output_data  = reinterpret_cast<char *>(output_blob->GetHandle().base);
    const int ele_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type); 
    auto count         = DimsVectorUtils::Count(input_dims);
    
    // Create Ordered, Positive shifts from param.shifts. 
    std::vector<int> shifts(input_dims.size(), 0);
    for (int d=0; d<roll_param->dims.size(); d++) {
        int dim = roll_param->dims[d];
        shifts[dim] = roll_param->shifts[d] < 0 ? roll_param->shifts[d] + input_dims[dim] : roll_param->shifts[d]; 
    }

    for (int i=0; i<count; i++) {
        // Too Many Calls of Memcpy is not a good choice here when speed matters.
        // Address of input, output should be different. Inplace Mode not supported.
        int out_i = 0;
        int remainder = i;
        for (int d=0; d<input_dims.size(); d++) {
            int stride_dim    = DimsVectorUtils::Count(input_dims, d+1);
            int in_index_dim  = remainder / stride_dim;
            int out_index_dim = (in_index_dim + shifts[d]) % input_dims[d];
            out_i += stride_dim * out_index_dim;
            remainder %= stride_dim;
        }
        memcpy(output_data + ele_size*out_i, input_data + ele_size*i, ele_size);
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(Roll, LAYER_ROLL);

}  // namespace TNN_NS
