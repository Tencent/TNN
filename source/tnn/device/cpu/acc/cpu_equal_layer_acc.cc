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
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

DECLARE_CPU_ACC(Equal, LAYER_EQUAL);

Status CpuEqualLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuEqualLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *output_blob = outputs[0];
    
    std::vector<void *> input_ptrs;
    std::vector<DimsVector> input_shapes;
    for (size_t inid = 0; inid < inputs.size(); inid++) {
        input_ptrs.push_back(inputs[inid]->GetHandle().base);
        input_shapes.push_back(inputs[inid]->GetBlobDesc().dims);
    }
    
    auto data_type = inputs[0]->GetBlobDesc().data_type;
    void *output_data = output_blob->GetHandle().base;
    const auto &output_dims = output_blob->GetBlobDesc().dims;
 
    if (data_type == DATA_TYPE_FLOAT) {
        CPU_ELEMENT_WISE_COMPARE<float, char>(input_ptrs, input_shapes, output_data, output_dims,
                                  [](float a, float b) -> char { return a == b; });
    } else if(data_type == DATA_TYPE_INT32) {  
        CPU_ELEMENT_WISE_COMPARE<int, char>(input_ptrs, input_shapes, output_data, output_dims,
                                  [](int a, int b) -> char { return a == b; });
    } else if(data_type == DATA_TYPE_INT8) {
        CPU_ELEMENT_WISE_COMPARE<char, char>(input_ptrs, input_shapes, output_data, output_dims,
                                  [](char a, char b) -> char { return a == b; });
    } else {
        LOGE("Error: CpuEqualLayerAcc don't support data type: %d\n", data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuEqualLayerAcc don't support data type");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Equal, LAYER_EQUAL);

}  // namespace TNN_NS
