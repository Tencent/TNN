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
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(Pad, LAYER_PAD)

Status NpuPadLayer::Convert() {
    // parameter and weight of the pad layer
    auto param = dynamic_cast<PadLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    // paddings
    std::vector<int> paddings = {
        0, 0, param->pads[4], param->pads[5], param->pads[2], param->pads[3], param->pads[0], param->pads[1]};
    std::shared_ptr<ge::op::Const> paddings_const = std::make_shared<ge::op::Const>(layer_name_ + "_paddings");
    ge::TensorDesc desc(hiai::Shape({4, 2}), hiai::FORMAT_NCHW, hiai::DT_INT32);
    NpuUtils::CreateAttrArray(paddings_const, paddings, desc, 8);
    weight_ops_.push_back(paddings_const);

    if (param->type == 0) {
        // values
        std::vector<float> const_val                     = {param->value};
        std::shared_ptr<ge::op::Const> const_val_const = std::make_shared<ge::op::Const>(layer_name_ + "_values");
        ge::TensorDesc const_desc(hiai::Shape({1}), hiai::FORMAT_NCHW, hiai::DT_FLOAT);
        NpuUtils::CreateAttrArray(const_val_const, const_val, const_desc, 1);
        weight_ops_.push_back(const_val_const);

        auto output = std::make_shared<hiai::op::PadV2>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_input_paddings(*paddings_const);
        output->set_input_constant_values(*const_val_const);
        ADD_OUTPUT_OP(output)
    } else if (param->type == 1) {
        auto output = std::make_shared<hiai::op::MirrorPad>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_input_paddings(*paddings_const);
        output->set_attr_mode("REFLECT");
        ADD_OUTPUT_OP(output)
    } else {
        LOGE("Npu does not support such padding\n");
        return Status(TNNERR_LAYER_ERR, "Npu does not support such padding");
    }
}

REGISTER_NPU_LAYER(Pad, LAYER_PAD)

}  // namespace TNN_NS