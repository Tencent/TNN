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

#include <cmath>
#include <memory>

#include <ngraph/node.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <inference_engine.hpp>

#include "tnn/layer/base_layer.h"
#include "tnn/device/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/device/openvino/openvino_types.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(HardSwish, LAYER_HARDSWISH);

Status HardSwishOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<HardSwishLayerParam*>(param_);

     if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes();
    
    std::shared_ptr<ngraph::Node> input_node0, input_node1;
    if (input_node.size() == 1) {
        input_node0 = input_node[0];
        input_node1 = input_node[0];
    } else {
        input_node0 = input_node[0];
        input_node1 = input_node[1];
    }

    // mul-add-clamp-mul
    auto shape = ngraph::Shape(input_node1->get_output_shape(0).size(), 1);

    auto mulConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::f32, shape, std::vector<float>{paramlist->alpha});
    auto mulNode = std::make_shared<ngraph::op::v1::Multiply>(input_node1->output(0), mulConst);
    mulNode->validate_and_infer_types();

    auto addConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::f32, shape, std::vector<float>{paramlist->beta});
    auto addNode = std::make_shared<ngraph::op::v1::Add>(mulNode->output(0), addConst);
    addNode->validate_and_infer_types();

    auto clampNode = std::make_shared<ngraph::op::Clamp>(
        addNode->output(0), 0.0f, 1.0f);
    clampNode->validate_and_infer_types();

    auto hardSwishNode = std::make_shared<ngraph::op::v1::Multiply>(
        input_node0->output(0), clampNode->output(0));

    hardSwishNode->validate_and_infer_types();
    hardSwishNode->set_friendly_name(paramlist->name);

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(hardSwishNode);
    SetOutputNodes(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(HardSwish, LAYER_HARDSWISH);

}