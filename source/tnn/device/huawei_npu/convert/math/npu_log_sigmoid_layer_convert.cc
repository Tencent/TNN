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

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(LogSigmoid, LAYER_LOGSIGMOID)

Status NpuLogSigmoidLayer::Convert() {
    auto sigmoid_op = std::make_shared<hiai::op::Activation>(layer_name_ + "_sigmoid");
    sigmoid_op->set_input_x(*input_ops_[0]->GetOperator());
    sigmoid_op->set_attr_mode(0);
    weight_ops_.push_back(sigmoid_op);

    auto output = std::make_shared<hiai::op::Log>(outputs_name_[0]);
    output->set_input_x(*sigmoid_op);
    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(LogSigmoid, LAYER_LOGSIGMOID);

}  // namespace TNN_NS
