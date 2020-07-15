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

#include <tnn/utils/data_type_utils.h>
#include "graph/attr_value.h"
#include "graph/op/nn_defs.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(InstanceNorm, LAYER_INST_BATCH_NORM);

Status NpuInstanceNormLayer::Convert() {
    auto layer_res = dynamic_cast<InstanceNormLayerResource*>(resource_);
    if (!layer_res) {
        LOGE("Error: InstanceNorm layer resource is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: InstanceNorm layer resource is nil");
    }
    // input shape nchw
    int input_channel = input_ops_[0]->GetShape()[1];
    // scale
    std::string scale_name = layer_name_ + "_scale";
    ge::Shape scale_shape({1, input_channel, 1, 1});
    auto scale = std::make_shared<ge::op::Const>(scale_name);
    NpuUtils::CreateAttrValue(scale, scale_shape, layer_res->scale_handle);
    weight_ops_.push_back(scale);

    // bias data
    std::string bias_name = layer_name_ + "_bias";
    ge::Shape bias_shape({1, input_channel, 1, 1});
    auto bias_const = std::make_shared<ge::op::Const>(bias_name);
    NpuUtils::CreateAttrValue(bias_const, bias_shape, layer_res->bias_handle);
    weight_ops_.push_back(bias_const);

    auto output = std::make_shared<ge::op::InstanceNorm>(outputs_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_input_scale(*scale);
    output->set_input_bias(*bias_const);
    output->set_attr_reduction_indices(ge::AttrValue::LIST_INT{1, 2});
    std::shared_ptr<OperatorInfo> output_op = std::make_shared<OperatorInfo>(output);
    output_ops_.push_back(output_op);
    return SetOutputOps();
}

REGISTER_NPU_LAYER(InstanceNorm, LAYER_INST_BATCH_NORM);

}  // namespace TNN_NS
