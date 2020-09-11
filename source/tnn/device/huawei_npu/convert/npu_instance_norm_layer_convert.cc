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
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(InstanceNorm, LAYER_INST_BATCH_NORM)

Status NpuInstanceNormLayer::Convert() {
    auto resource = dynamic_cast<InstanceNormLayerResource*>(resource_);
    if (!resource) {
        return Status(TNNERR_MODEL_ERR, "Error: InstanceNorm layer resource is nil");
    }
    // input shape 0.n 1.c 2.h 3.w
    int input_channel = input_ops_[0]->GetShape()[1];
    // scale
    std::string scale_name = layer_name_ + "_scale";
    ge::Shape scale_shape({1, input_channel, 1, 1});
    auto scale = std::make_shared<ge::op::Const>(scale_name);
    NpuUtils::CreateAttrValue(scale, scale_shape, resource->scale_handle);
    weight_ops_.push_back(scale);

    // bias data
    std::string bias_name = layer_name_ + "_bias";
    ge::Shape bias_shape({1, input_channel, 1, 1});
    auto bias_const = std::make_shared<ge::op::Const>(bias_name);
    NpuUtils::CreateAttrValue(bias_const, bias_shape, resource->bias_handle);
    weight_ops_.push_back(bias_const);

    auto output = std::make_shared<hiai::op::InstanceNorm>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_input_gamma(*scale);
    output->set_input_beta(*bias_const);

    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(InstanceNorm, LAYER_INST_BATCH_NORM)

}  // namespace TNN_NS
