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

DECLARE_NPU_LAYER(Concat, LAYER_CONCAT)

Status NpuConcatLayer::Convert() {
    int input_size = input_ops_.size();
    auto param     = dynamic_cast<ConcatLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    if (input_size < 2) {
        return Status(TNNERR_LAYER_ERR, "Concat layer's inputs size must >= 2");
    }
    int axis = param->axis;

    auto output = std::make_shared<ge::op::Concat>(outputs_name_[0]);
    // multiple input
    output->create_dynamic_input_x(input_size);
    for (int i = 1; i < input_size + 1; i++) {
        output->set_dynamic_input_x(i, *(input_ops_[i - 1]->GetOperator()));
    }
    output->set_attr_axis(axis);
    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(Concat, LAYER_CONCAT)

}  // namespace TNN_NS
