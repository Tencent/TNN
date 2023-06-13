// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "tnn/network/openvino/custom_layer/custom_softsign.h"
#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(Softsign, LAYER_SOFTSIGN);

Status SoftsignOVLayerBuilder::Build() {

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }

    auto input_node = GetInputNodes();
    ngraph::OutputVector inputs;
    for (auto item : input_node) {
        inputs.push_back(item->output(0));
    }
   
    auto softsignNode = std::make_shared<CustomSoftsignOp>(
        inputs, base_layer_, GetInputBlobs(), GetOutputBlobs());

    softsignNode->validate_and_infer_types();
    softsignNode->set_friendly_name(param_->name);

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(softsignNode);
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Softsign, LAYER_SOFTSIGN);

}
