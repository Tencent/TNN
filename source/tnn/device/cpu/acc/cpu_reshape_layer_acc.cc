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
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC_WITH_FUNC(Reshape, LAYER_RESHAPE,
                          virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
                                                                 const std::vector<Blob *> &outputs););

Status CpuReshapeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuReshapeLayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto *layer_param = dynamic_cast<ReshapeLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    Status status = TNN_OK;
    auto input_dims = inputs[0]->GetBlobDesc().dims;
    if (inputs.size() >= 2) {
        if (inputs[1]->GetBlobDesc().data_type != DATA_TYPE_INT32) {
            return Status(TNNERR_PARAM_ERR, "Reshape input(shape) has invalid data type");
        }
        
        auto dim_count = DimsVectorUtils::Count(inputs[1]->GetBlobDesc().dims);
        auto dim_data = (int *)((char *)inputs[1]->GetHandle().base + inputs[1]->GetHandle().bytes_offset);
        DimsVector dims;
        for (int i=0; i<dim_count; i++) {
            dims.push_back(dim_data[i]);
        }
        layer_param->shape = dims;
        layer_param->num_axes = dim_count;
        auto output_dims = DimsVectorUtils::Reshape(input_dims, dims, layer_param->axis, dim_count, &status);
        RETURN_ON_NEQ(status, TNN_OK);
        
        outputs[0]->GetBlobDesc().dims = output_dims;
    }
    
    //Adjust params to diffrent batch\height\width with 0 and -1
    auto shape = layer_param->shape;
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    if (shape.size() == output_dims.size()) {
        const auto count = MIN(output_dims.size(), input_dims.size());
        
        //reset 0
        {
            for (auto i=0; i<count; i++) {
                if (output_dims[i]> 0 && input_dims[i] == output_dims[i] && shape[i] != -1) {
                    shape[i] = 0;
                }
            }
        }
        
        //reset -1
        {
            int non_zero_index = -1;
            int non_zero_count = 0;
            for (auto i=0; i<shape.size(); i++) {
                if (shape[i] != 0) {
                    non_zero_index = i;
                    non_zero_count++;
                }
            }
            
            if (non_zero_count == 1) {
                shape[non_zero_index] = -1;
            }
        }
        
        auto infer_output_dims = DimsVectorUtils::Reshape(input_dims, shape, layer_param->axis, (int)shape.size(), &status);
        if (status == TNN_OK && DimsVectorUtils::Equal(infer_output_dims, output_dims)) {
            layer_param->shape = shape;
        }
    }
    
    return TNN_OK;
}

Status CpuReshapeLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto &input  = inputs[0];
    auto &output = outputs[0];
    auto param   = (ReshapeLayerParam *)param_;
    ASSERT(param != nullptr);
    if (param->reshape_type == 0) {
        if (output->GetHandle().base != input->GetHandle().base) {
            auto dims_input    = input->GetBlobDesc().dims;
            int data_byte_size = DataTypeUtils::GetBytesSize(output->GetBlobDesc().data_type);
            auto size_in_bytes = DimsVectorUtils::Count(dims_input) * data_byte_size;
            memcpy(output->GetHandle().base, input->GetHandle().base, size_in_bytes);
        }
    } else if (param->reshape_type == 1) {
        // tensorflow reshape
        DataFormatConverter::ConvertFromNCHWToNHWC<float>(input, output);
        DataFormatConverter::ConvertFromNHWCToNCHW<float>(output, nullptr);
    } else {
        LOGE("Error: Unsupport reshape type(%d)", param->reshape_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuReshapeLayerAcc failed!\n");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Reshape, LAYER_RESHAPE);

}  // namespace TNN_NS
