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

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_X86_ACC(Gather, LAYER_GATHER);

Status X86GatherLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<GatherLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);
    int axis = layer_param->axis;
    
    auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);
    if ((layer_param->data_in_resource || layer_param->indices_in_resource) && !layer_resource) {
        return Status(TNNERR_MODEL_ERR, "Gather resource is invalid");
    }
    
    DimsVector input_data_dims;
    char *input_data_ptr = nullptr;
    if (layer_param->data_in_resource) {
        input_data_dims = layer_resource->data.GetBufferDims();
        input_data_ptr = layer_resource->data.force_to<char*>();
    } else {
        input_data_dims = (*(inputs.begin()))->GetBlobDesc().dims;
        input_data_ptr = handle_ptr<char*>((*(inputs.begin()))->GetHandle());
    }
    
    DimsVector indices_dims;
    int *indices_data_ptr = nullptr;
    if (layer_param->indices_in_resource) {
        indices_dims = layer_resource->indices.GetBufferDims();
        indices_data_ptr = layer_resource->indices.force_to<int*>();
    } else {
        indices_dims = (*(inputs.rbegin()))->GetBlobDesc().dims;
        indices_data_ptr = handle_ptr<int *>((*(inputs.rbegin()))->GetHandle());
    }
    
    const int slice_size = DimsVectorUtils::Count(input_data_dims, axis+1);
    const int input_slice_count = DimsVectorUtils::Count(input_data_dims, axis, axis+1);
    const int batch = DimsVectorUtils::Count(input_data_dims, 0, axis);
    
    const auto output_dims = outputs[0]->GetBlobDesc().dims;
    const int output_slice_count = DimsVectorUtils::Count(indices_dims);
    
    const int ele_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);
    auto output_data_ptr = handle_ptr<char*>(outputs[0]->GetHandle());
    
    for (int b=0; b<batch; b++) {
        int input_index_b = b*input_slice_count*slice_size;
        int output_index_b = b*output_slice_count*slice_size;
        for (int i=0; i<output_slice_count; i++) {
            int slice_index = indices_data_ptr[i];
            if (slice_index < 0 || slice_index >= input_slice_count) {
                LOGE("X86GatherLayerAcc::Forward invalid slice_index\n");
                return Status(TNNERR_MODEL_ERR, "X86GatherLayerAcc::Forward invalid slice_index");
            }
            int input_index = input_index_b + slice_index*slice_size;
            int output_index = output_index_b + i*slice_size;
            
            memcpy(output_data_ptr + output_index*ele_size,
                   input_data_ptr + input_index*ele_size,
                   slice_size * ele_size);
        }
    }
    return TNN_OK;
}

REGISTER_X86_ACC(Gather, LAYER_GATHER);
}  // namespace TNN_NS
