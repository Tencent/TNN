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

#include <graph/op/all_ops.h>
#include <tnn/core/status.h>
#include "tnn/device/huawei_npu/convert/npu_base_layer_convert.h"
#include "tnn/device/huawei_npu/convert/npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(Mul, LAYER_MUL)

Status NpuMulLayer::Convert() {
    auto param    = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    auto resource = dynamic_cast<EltwiseLayerResource *>(resource_);
    CHECK_PARAM_NULL(param);

    int input_size = input_ops_.size();
    if (!((input_size == 1 && resource) || input_size == 2)) {
        return Status(TNNERR_MODEL_ERR, "Error: the Multiply layer input number is not correct");
    }

    auto output   = std::make_shared<ge::op::Mul>(outputs_name_[0]);

    if (input_size == 2) {
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_input_y(*input_ops_[1]->GetOperator());
    } else {
        auto weight_const             = std::make_shared<ge::op::Const>(layer_name_ + "_weight");
        std::vector<int> weight_shape = resource->element_shape;
        std::vector<int> input_shape  = input_ops_[0]->GetShape();
        Status calculate_ret          = NpuUtils::CalculateBroadcastSize(weight_shape, resource, input_shape);
        if (calculate_ret != TNN_OK) {
            return calculate_ret;
        }
        ge::Shape weight_shape_op({weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]});
        NpuUtils::CreateAttrValue(weight_const, weight_shape_op, resource->element_handle);
        weight_ops_.push_back(weight_const);

        output->set_input_x(*weight_const);
        output->set_input_y(*input_ops_[0]->GetOperator());
    }

    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(Mul, LAYER_MUL)

}  // namespace TNN_NS
