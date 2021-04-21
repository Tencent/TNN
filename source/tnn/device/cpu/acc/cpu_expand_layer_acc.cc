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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC_WITH_FUNC(Expand, LAYER_EXPAND,
                          virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs););

Status CpuExpandLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuExpandLayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto expand_param = dynamic_cast<ExpandLayerParam*>(param_);
    CHECK_PARAM_NULL(expand_param);
    
    if (inputs.size() == 2) {
        auto data_dims = inputs[0]->GetBlobDesc().dims;
        DimsVector shape_dims;
        auto shape_data = (int *)inputs[1]->GetHandle().base;
        auto shape_data_count = DimsVectorUtils::Count(inputs[1]->GetBlobDesc().dims);
        for (int i=0; i<shape_data_count; i++) {
            shape_dims.push_back(shape_data[i]);
        }
        
        expand_param->shape = shape_dims;
        
        auto output_dims = DimsFunctionUtils::Expand(data_dims, shape_dims, nullptr);
        outputs[0]->GetBlobDesc().dims = output_dims;
    }
    
    return AbstractLayerAcc::InferRuntimeOutputShape(inputs, outputs);
}

Status CpuExpandLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];
    auto output_dims = output_blob->GetBlobDesc().dims;
    auto input_dims = input_blob->GetBlobDesc().dims;
    
    const int ele_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);
    
    
    int diff = output_dims.size() - input_dims.size();
    
    char *input_data  = reinterpret_cast<char *>(input_blob->GetHandle().base);
    char *output_data = reinterpret_cast<char *>(output_blob->GetHandle().base);
    int output_diff_start_cnt = DimsVectorUtils::Count(output_dims, diff);
    for(int i = 0; i < output_diff_start_cnt; ++i) {
        int index = i, in_index = 0;
        for(int j = input_dims.size() - 1; j >= 0; --j) {
            int input_dim = input_dims[j];
            int output_dim = output_dims[j + diff];
            int mod = index % output_dim;
            if(input_dim == 1) {
                mod = 0;
            }
            index /= output_dim;
            in_index += mod * DimsVectorUtils::Count(input_dims, j + 1);
        }
        //output_data[i] = input_data[in_index];
        memcpy(output_data + i*ele_size, input_data + in_index*ele_size, ele_size);
    }
    if(diff > 0) {
        const int data_size = output_diff_start_cnt * ele_size;
        int repeat_cnt      = DimsVectorUtils::Count(output_dims, 0, diff);
        for(int i = 1; i < repeat_cnt; ++i) {
            memcpy(output_data + i * data_size, output_data, data_size);
        }
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Expand, LAYER_EXPAND);

}  // namespace TNN_NS
