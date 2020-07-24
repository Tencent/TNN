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
#include "graph/op/all_ops.h"
#include "graph/op/nn_defs.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(InnerProduct, LAYER_INNER_PRODUCT)

Status NpuInnerProductLayer::Convert() {
    auto param    = dynamic_cast<InnerProductLayerParam *>(param_);
    auto resource = dynamic_cast<InnerProductLayerResource *>(resource_);
    CHECK_PARAM_NULL(param);
    if (!resource) {
        return Status(TNNERR_MODEL_ERR, "Error: InnerProductLayerResource is nil");
    }

    // weight
    vector<int> input_shape = input_ops_[0]->GetShape();
    std::string weight_name = layer_name_ + "_weight";
    ge::Shape weight_shape({param->num_output, input_shape[1], 1, 1});
    auto weight_const = std::make_shared<ge::op::Const>(weight_name);
    NpuUtils::CreateAttrValue(weight_const, weight_shape, resource->weight_handle);
    weight_ops_.push_back(weight_const);

    // bias
    auto output = std::make_shared<ge::op::FullConnection>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_input_w(*weight_const);
    int bias_count = resource->bias_handle.GetDataCount();
    if (param->has_bias) {
        std::string bias_name = layer_name_ + "_bias";
        ge::Shape bias_shape({1, bias_count, 1, 1});
        auto bias_const = std::make_shared<ge::op::Const>(bias_name);
        NpuUtils::CreateAttrValue(bias_const, bias_shape, resource->bias_handle);
        weight_ops_.push_back(bias_const);
        output->set_input_b(*bias_const);
    }
    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(InnerProduct, LAYER_INNER_PRODUCT)

}  // namespace TNN_NS
