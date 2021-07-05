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

#include "cpu_layer_acc.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC_WITH_FUNC(NonZero, LAYER_NONZERO,
                          virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
                          void PutDataIndex(int *dst, const DimsVector stride, const DimsVector index););

Status CpuNonZeroLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

void CpuNonZeroLayerAcc::PutDataIndex(int *dst, const DimsVector stride, const DimsVector index) {
    for (int i=0; i<index.size(); i++) {
        *dst = index[i];
        dst += stride[1];
    }
}

Status CpuNonZeroLayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto input_count = DimsVectorUtils::Count(input_dims);
    auto ele_size = DataTypeUtils::GetBytesSize(inputs[0]->GetBlobDesc().data_type);
    
    auto input_data_ptr = (char *)inputs[0]->GetHandle().base;
    
    //runtime compute count
    int nonzero_count = 0;
    for (int index=0; index<input_count; index++) {
        bool is_non_zero = false;
        for (int i=0; i<ele_size; i++) {
            if (input_data_ptr[i] != 0) {
                is_non_zero = true;
                break;
            }
        }
        
        if (is_non_zero) {
            nonzero_count++;
        }
        
        input_data_ptr += ele_size;
    }
    
    int input_dim_size = (int)input_dims.size();
    outputs[0]->GetBlobDesc().dims = {input_dim_size, nonzero_count};
    
    return AbstractLayerAcc::InferRuntimeOutputShape(inputs, outputs);
}

Status CpuNonZeroLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto input_count = DimsVectorUtils::Count(input_dims);
    auto ele_size = DataTypeUtils::GetBytesSize(inputs[0]->GetBlobDesc().data_type);
    
    auto output_blob  = outputs[0];
    auto output_dims = output_blob->GetBlobDesc().dims;
    auto output_shtride = DimsFunctionUtils::StrideOfShape(output_dims);
    
    auto input_data_ptr = (char *)inputs[0]->GetHandle().base;
    int *output_data_ptr = (int *)outputs[0]->GetHandle().base;
    
    DimsVector dim_index(input_dims.size(), 0);
    for (int index=0; index<input_count; index++) {
        bool is_non_zero = false;
        for (int i=0; i<ele_size; i++) {
            if (input_data_ptr[i] != 0) {
                is_non_zero = true;
                break;
            }
        }
        
        if (is_non_zero) {
            PutDataIndex(output_data_ptr, output_shtride, dim_index);
            output_data_ptr++;
        }
        
        input_data_ptr += ele_size;
        dim_index = DimsFunctionUtils::IncreaseIndex(dim_index, input_dims);
    }
    
    
    return TNN_OK;
}

REGISTER_CPU_ACC(NonZero, LAYER_NONZERO);
}  // namespace TNN_NS
