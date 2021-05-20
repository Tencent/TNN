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
namespace TNN_NS {

DECLARE_CPU_ACC(ConstantOfShape, LAYER_CONSTANT_OF_SHAPE);

Status CpuConstantOfShapeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuConstantOfShapeLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_resource = dynamic_cast<ConstantOfShapeLayerResource*>(resource_);
    CHECK_PARAM_NULL(layer_resource);
    auto data_value_count = layer_resource->value.GetDataCount();
    auto data_value_size = layer_resource->value.GetBytesSize();
    auto data_value_ptr = layer_resource->value.force_to<char *>();
    
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    auto output_count = DimsVectorUtils::Count(output_dims);
    auto output_data_ptr = (char *)outputs[0]->GetHandle().base;
    
    //support the case if constofshape has empty output blob with dims={0}
    if (output_dims.size() == 1 && output_dims[0] == 0) {
        return TNN_OK;
    }
    
    if (output_dims.size() <= 0 || output_data_ptr==nullptr || output_count <= 0) {
        return Status(TNNERR_LAYER_ERR, "ConstantOfShape has invalid param or resource");
    }

    for (int i=0; i<output_count; i++) {
        memcpy(output_data_ptr + i*data_value_size, data_value_ptr, data_value_size);
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(ConstantOfShape, LAYER_CONSTANT_OF_SHAPE);
}  // namespace TNN_NS
