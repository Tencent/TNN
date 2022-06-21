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

#include "tnn/network/openvino/layer_builder/adapter_layer_builder.h"

#include "tnn/network/openvino/custom_layer/custom_implementation.h"
#include "tnn/network/openvino/utils.h"

namespace TNN_NS {

DECLARE_CUSTOM_OP(Adapter);
REGISTER_CUSTOM_OP(Adapter);

DECLARE_CUSTOM_IMPLEMENTATION(Adapter);
REGISTER_CUSTOM_IMPLEMENTATION(Adapter, CustomAdapter);

Status AdapterOVLayerBuilder::Build() {
    std::shared_ptr<ngraph::Node> customNode;
    auto input_nodes = GetInputNodes();

    if (input_nodes.size() <= 0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }

    if (GetOutputBlobs().size() != 1) {
        LOGE("Error: ov adapter layer outputs must be single");
        return TNNERR_INIT_LAYER;
    }

    ngraph::OutputVector inputs;
    for (auto item : input_nodes) {
        inputs.push_back(item->output(0));
    }

    customNode = std::make_shared<CustomAdapterOp>(inputs, base_layer_, GetInputBlobs(), GetOutputBlobs());
    customNode->set_friendly_name(param_->name);
    customNode->validate_and_infer_types();
    ngraph::NodeVector outputNodes;
    outputNodes.push_back(customNode);
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

}  // namespace TNN_NS
