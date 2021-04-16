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
#include "tnn/device/huawei_npu/convert/npu_base_layer_convert.h"
#include "tnn/device/huawei_npu/convert/npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(Hardswish, LAYER_HARDSWISH)

Status NpuHardswishLayer::Convert() {
    auto param = dynamic_cast<HardSwishLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    if (!(param->alpha >= 0.1666f && param->alpha <= 0.1667f && param->beta >= 0.4999f && param->beta <= 0.5001f)) {
        LOGE("hardswish only support alpha=1/6 beta=0.5, but in fact, alpha=%f beta=%f\n", param->alpha, param->beta);
        return Status(TNNERR_LAYER_ERR, "Error: Npu currently only supports hardswish (alpha=1/6, beta=0.5)");
    }

    int input_size = input_ops_.size();
    if (input_size == 1) {
        auto output = std::make_shared<hiai::op::HardSwish>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        ADD_OUTPUT_OP(output)
    } else if (input_size == 2) {
        // hardswish will be broken into hardsigmoid+mul
        auto sub_op = std::make_shared<ge::op::Activation>(layer_name_ + "_sigmoid");
        sub_op->set_input_x(*input_ops_[1]->GetOperator());
        sub_op->set_attr_mode(10);
        weight_ops_.push_back(sub_op);

        auto output = std::make_shared<hiai::op::Mul>(outputs_name_[0]);
        output->set_input_x1(*input_ops_[0]->GetOperator());
        output->set_input_x2(*sub_op);
        ADD_OUTPUT_OP(output)
    } else {
        printf("the Unary input size is not correct\n");
        return Status(TNNERR_PARAM_ERR, "Error: the Unary layer count is not correct");
    }
}

REGISTER_NPU_LAYER(Hardswish, LAYER_HARDSWISH);

}  // namespace TNN_NS
