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
    int input_dims_size = (int)input_ops_[0]->GetShape().size();
    vector<int> w_shape = input_ops_[0]->GetShape();
    w_shape[0]          = param->num_output;
    for (int i = input_dims_size; i < 4; ++i) {
        w_shape.push_back(1);
    }
    ge::Shape weight_shape(NpuUtils::Int32VecToTVec<int64_t>(w_shape));
    auto weight_const = std::make_shared<ge::op::Const>(layer_name_ + "_weight");
    NpuUtils::CreateAttrValue(weight_const, weight_shape, resource->weight_handle);
    weight_ops_.push_back(weight_const);

    auto output = std::make_shared<hiai::op::FullyConnection>(outputs_name_[0]);
    if (input_dims_size < 4) {
        // insert Reshape layer if input dims size < 4
        std::vector<int> shape;
        shape.clear();
        for (int i = 0; i < input_dims_size; ++i) {
            shape.push_back(0);
        }
        for (int i = input_dims_size; i < 4; ++i) {
            shape.push_back(1);
        }
        std::shared_ptr<ge::op::Const> shape_const = std::make_shared<ge::op::Const>(layer_name_ + "_reshape_shape");
        ge::TensorDesc shape_desc(ge::Shape({(int64_t)shape.size()}), ge::FORMAT_NCHW, ge::DT_INT32);
        NpuUtils::CreateAttrArray(shape_const, shape, shape_desc, (int)shape.size());
        weight_ops_.push_back(shape_const);

        auto reshape_op = std::make_shared<hiai::op::Reshape>(layer_name_ + "_reshape");
        reshape_op->set_input_x(*input_ops_[0]->GetOperator());
        reshape_op->set_input_shape(*shape_const);
        weight_ops_.push_back(reshape_op);

        output->set_input_x(*reshape_op);
    } else {
        output->set_input_x(*input_ops_[0]->GetOperator());
    }
    output->set_input_w(*weight_const);
    output->set_attr_num_output(param->num_output);
    // bias
    if (param->has_bias) {
        std::string bias_name = layer_name_ + "_bias";
        int bias_count        = resource->bias_handle.GetDataCount();
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
