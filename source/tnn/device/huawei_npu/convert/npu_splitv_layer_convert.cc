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

DECLARE_NPU_LAYER_WEIGHT(SplitV, LAYER_SPLITV)

Status NpuSplitVLayer::Convert() {
    auto param = dynamic_cast<SplitVLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    std::vector<int> slices_vec = param->slices;
    ge::TensorDesc slices_desc(ge::Shape({(int64_t)slices_vec.size()}), ge::FORMAT_NCHW, ge::DT_INT32);
    std::shared_ptr<ge::op::Const> slices_const = std::make_shared<ge::op::Const>(layer_name_ + "_slices_const");
    RETURN_ON_NEQ(NpuUtils::CreateAttrArray(slices_const, slices_vec, slices_desc, slices_vec.size()), TNN_OK);
    weight_ops_.push_back(slices_const);

    std::vector<int> split_dim_vec = {param->axis};
    ge::TensorDesc split_dim_desc(ge::Shape({1}), ge::FORMAT_NCHW, ge::DT_INT32);
    std::shared_ptr<ge::op::Const> split_dim_const = std::make_shared<ge::op::Const>(layer_name_ + "_split_dim_const");
    RETURN_ON_NEQ(NpuUtils::CreateAttrArray(split_dim_const, split_dim_vec, split_dim_desc, 1), TNN_OK);
    weight_ops_.push_back(split_dim_const);

    std::string split_output_name = layer_name_ + "split_output";
    auto output = std::make_shared<hiai::op::SplitV>(split_output_name);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_input_size_splits(*slices_const);
    output->set_input_split_dim(*split_dim_const);
    output->set_attr_num_split(param->slices.size());
    output->create_dynamic_output_y(param->slices.size());
    weight_ops_.push_back(output);

    // create data op to split output
    // get output ops
    for (int i = 0; i < outputs_name_.size(); i++) {
        auto temp_op = std::make_shared<hiai::op::Permute>(outputs_name_[i]);
        std::string output_node_name = "y" + std::to_string(i+1);

        std::vector<int64_t> order;
        order.clear();
        for (int idx = 0; idx < output_shapes_[i].size(); ++idx) {
            order.push_back(idx);
        }

        temp_op->set_input_x(*output, output_node_name);
        temp_op->set_attr_order(order);

        std::shared_ptr<OperatorInfo> output_op = std::make_shared<OperatorInfo>(temp_op);
        output_op->SetShape(output_shapes_[i]);
        output_ops_.push_back(output_op);
    }

    return TNN_OK;
}

REGISTER_NPU_LAYER(SplitV, LAYER_SPLITV)

}  // namespace TNN_NS
