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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC_WITH_FUNC(OneHot, LAYER_ONEHOT,
                          virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
                                                                 const std::vector<Blob *> &outputs););

Status CpuOneHotLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuOneHotLayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto *layer_param = dynamic_cast<OneHotLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (inputs.size() >= 3) {
        //depth
        {
            auto data = (int *)((char *)inputs[1]->GetHandle().base + inputs[1]->GetHandle().bytes_offset);
            if (inputs[1]->GetBlobDesc().data_type != DATA_TYPE_INT32) {
                return Status(TNNERR_PARAM_ERR, "OneHotLayer has invalid layer resource for depth param");
            }
            layer_param->depth = data[0];
        }
        
        //values
        {
            auto data = (float *)((char *)inputs[2]->GetHandle().base + inputs[2]->GetHandle().bytes_offset);
            if (inputs[2]->GetBlobDesc().data_type!= DATA_TYPE_FLOAT ||
                DimsVectorUtils::Count(inputs[2]->GetBlobDesc().dims) < 2) {
                return Status(TNNERR_PARAM_ERR, "OneHotLayer has invalid layer resource for values param");
            }
            layer_param->value_off = data[0];
            layer_param->value_on = data[1];
        }
        
        //infer output shape
        int axis = layer_param->axis;
        auto output_dims = inputs[0]->GetBlobDesc().dims;
        if (axis < 0) {
            axis += output_dims.size() + 1;
        }
        
        output_dims.insert(output_dims.begin()+axis, layer_param->depth);
        
        outputs[0]->GetBlobDesc().dims = output_dims;
    }
    
    return TNN_OK;
}

Status CpuOneHotLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto *layer_param = dynamic_cast<OneHotLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    int axis = layer_param->axis;
    const auto input_dims = inputs[0]->GetBlobDesc().dims;
    const auto output_dims = outputs[0]->GetBlobDesc().dims;
    if (axis < 0) {
        axis += input_dims.size() + 1;
    }
    
    //see https://github.com/onnx/onnx/blob/master/docs/Operators.md#OneHot
    auto input_data_type  = inputs[0]->GetBlobDesc().data_type;
    RETURN_VALUE_ON_NEQ(input_data_type, DATA_TYPE_INT32,
                        Status(TNNERR_MODEL_ERR, "OneHot input indices must be INT"));
    auto output_data_type  = outputs[0]->GetBlobDesc().data_type;
    RETURN_VALUE_ON_NEQ(output_data_type, DATA_TYPE_FLOAT,
                        Status(TNNERR_MODEL_ERR, "OneHot only supports output with FLOAT"));
    
    auto input_data = (int *)((char *)inputs[0]->GetHandle().base + inputs[0]->GetHandle().bytes_offset);
    auto output_data = (float *)((char *)outputs[0]->GetHandle().base + outputs[0]->GetHandle().bytes_offset);

    const int output_count = DimsVectorUtils::Count(output_dims);
    
    auto value_off = layer_param->value_off;
    auto value_on = layer_param->value_on;
    
    for (int i = 0; i < output_count; ++i) {
        output_data[i] = value_off;
    }
    
    DimsVector input_index(input_dims.size(), 0);
    const int input_count = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims);
    for (int i = 0; i < input_count; ++i) {
        int depth = input_data[i];
        auto output_index = input_index;
        output_index.insert(output_index.begin()+axis, depth);
        auto output_offset = DimsOffsetUtils::ConvertIndexToOffset(output_dims, output_index);
        output_data[output_offset] = value_on;
        
        input_index = DimsFunctionUtils::IncreaseIndex(input_index, input_dims);
    }
    
    return TNN_OK;
}

REGISTER_CPU_ACC(OneHot, LAYER_ONEHOT);
}  // namespace TNN_NS
