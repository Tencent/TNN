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

#include <algorithm>
#include <cmath>

#include "cpu_binary_op_layer_acc.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {
DECLARE_CPU_ACC(Where, LAYER_WHERE);

Status CpuWhereLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuWhereLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    //X, Y, condition order for input
    Blob *output_blob = outputs[0];
    
    std::vector<void *> input_ptrs;
    std::vector<DimsVector> input_shapes;
    // if x, y in resource, push back x and y first.
    WhereLayerResource* layer_resource = dynamic_cast<WhereLayerResource *>(resource_);
    if (layer_resource) {
        if (layer_resource->x.GetBytesSize()>0) {
            auto dims = layer_resource->x.GetBufferDims();
            input_ptrs.push_back(layer_resource->x.force_to<int*>());
            input_shapes.push_back(dims);
        }
        if (layer_resource->y.GetBytesSize()>0) {
            auto dims = layer_resource->y.GetBufferDims();
            input_ptrs.push_back(layer_resource->y.force_to<int*>());
            input_shapes.push_back(dims);
        }
    }
    
    for (size_t inid = 0; inid < inputs.size(); inid++) {
        input_ptrs.push_back(inputs[inid]->GetHandle().base);
        input_shapes.push_back(inputs[inid]->GetBlobDesc().dims);
    }

    DataType condition_dtype = inputs[inputs.size()-1]->GetBlobDesc().data_type;
    DataType output_dtype    = output_blob->GetBlobDesc().data_type;
    if (output_dtype == DATA_TYPE_FLOAT || output_dtype == DATA_TYPE_INT32) {
        void *output_data = output_blob->GetHandle().base;
        const auto &output_dims = output_blob->GetBlobDesc().dims;
        if (condition_dtype == DATA_TYPE_INT32) {
            CPU_ELEMENT_WISE<int, int, int, int>(input_ptrs, input_shapes, output_data, output_dims,
                [](int a, int b, int c) -> int { return c!=0 ? a : b; });
        } else if (condition_dtype == DATA_TYPE_INT8) {
            CPU_ELEMENT_WISE<int, int, char, int>(input_ptrs, input_shapes, output_data, output_dims,
                [](int a, int b, char c) -> int { return c!=0 ? a : b; });
        } else {
            LOGE("Error: CpuWhereLayerAcc Condition don't support data type: %d\n", condition_dtype);
            return Status(TNNERR_MODEL_ERR, "Error: CpuEqualLayerAcc don't support your input condition data type");
        }
    } else if (output_dtype == DATA_TYPE_HALF || output_dtype == DATA_TYPE_BFP16) {
        int16_t *output_data = reinterpret_cast<int16_t*>(output_blob->GetHandle().base);
        const auto &output_dims = output_blob->GetBlobDesc().dims;
        if (condition_dtype == DATA_TYPE_INT32) {
            CPU_ELEMENT_WISE<int16_t, int16_t, int, int16_t>(input_ptrs, input_shapes, output_data, output_dims,
                [](int16_t a, int16_t b, int c) -> int16_t { return c!=0 ? a : b; });
        } else if (condition_dtype == DATA_TYPE_INT8) {
            CPU_ELEMENT_WISE<int16_t, int16_t, char, int16_t>(input_ptrs, input_shapes, output_data, output_dims,
                [](int16_t a, int16_t b, char c) -> int16_t { return c!=0 ? a : b; });
        } else {
            LOGE("Error: CpuWhereLayerAcc Condition don't support data type: %d\n", condition_dtype);
            return Status(TNNERR_MODEL_ERR, "Error: CpuEqualLayerAcc don't support your input condition data type");
        } 
    } else {
        LOGE("Error: CpuEqualLayerAcc don't support data type: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuEqualLayerAcc don't support data type");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Where, LAYER_WHERE);

}  // namespace TNN_NS
