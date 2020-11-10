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

DECLARE_NPU_LAYER_WEIGHT(Clip, LAYER_CLIP)

Status NpuClipLayer::Convert() {
    auto param = dynamic_cast<ClipLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    std::vector<float> min = {param->min};
    std::vector<float> max = {param->max};

    ge::TensorDesc desc({}, ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::AttrValue::TENSOR input_size_tensor = std::make_shared<ge::Tensor>(desc);
    auto min_const = std::make_shared<ge::op::Const>(layer_name_ + "_min");
    NpuUtils::CreateAttrArray(min_const, min, desc, 1);
    auto max_const = std::make_shared<ge::op::Const>(layer_name_ + "_max");
    NpuUtils::CreateAttrArray(max_const, max, desc, 1);

    weight_ops_.push_back(min_const);
    weight_ops_.push_back(max_const);
    auto output = std::make_shared<hiai::op::ClipByValue>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_input_clip_value_max(*max_const);
    output->set_input_clip_value_min(*min_const);
    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(Clip, LAYER_CLIP)

}  // namespace TNN_NS
