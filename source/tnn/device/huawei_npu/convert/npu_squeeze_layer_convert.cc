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

#include "graph/attr_value.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(Squeeze, LAYER_SQUEEZE)

Status NpuSqueezeLayer::Convert() {
    auto param = dynamic_cast<SqueezeLayerParam*>(param_);
    CHECK_PARAM_NULL(param);
    auto resource = dynamic_cast<SqueezeLayerResource*>(resource_);
    if (param->data_in_resource && !resource) {
        return Status(TNNERR_MODEL_ERR, "Squeeze resource is invalid");
    }

    auto output = std::make_shared<hiai::op::Squeeze>(outputs_name_[0]);
    if (param->data_in_resource) {
        DimsVector data_dims                      = resource->data.GetBufferDims();
        std::shared_ptr<ge::op::Const> data_const = std::make_shared<ge::op::Const>(layer_name_ + "_data");
        ge::Shape data_shape(NpuUtils::Int32VecToTVec<int64_t>(data_dims));
        NpuUtils::CreateAttrValue(data_const, data_shape, resource->data);
        weight_ops_.push_back(data_const);
        output->set_input_x(*data_const);
    } else {
        output->set_input_x(*input_ops_[0]->GetOperator());
    }
    output->set_attr_axis(ge::AttrValue::LIST_INT(NpuUtils::Int32VecToTVec<long>(param->axes)));
    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(Squeeze, LAYER_SQUEEZE)

}  // namespace TNN_NS
