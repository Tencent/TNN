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
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <inference_engine.hpp>

#include "tnn/layer/base_layer.h"
#include "tnn/device/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/device/openvino/openvino_types.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(Selu, LAYER_SELU);

Status SeluOVLayerBuilder::Build() {
    
    auto paramlist = dynamic_cast<SeluLayerParam*>(param_);

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    // {1,1,1,1} alpha gamma
    auto alphaNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::f32, ngraph::Shape(input_node->get_output_shape(0).size(), 1), \
        std::vector<float>{paramlist->alpha});
    auto gammaNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::f32, ngraph::Shape(input_node->get_output_shape(0).size(), 1), \
        std::vector<float>{paramlist->gamma});
    auto zeroNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::f32, ngraph::Shape(input_node->get_output_shape(0).size(), 1), \
        std::vector<float>{0.0f});

    auto posNode = std::make_shared<ngraph::op::v1::Maximum>(input_node->output(0), zeroNode);
    auto negNode = std::make_shared<ngraph::op::v1::Minimum>(input_node->output(0), zeroNode);

    // exp(negNode) * alpha - alpha
    auto powNode = std::make_shared<ngraph::op::Exp>(negNode);
    auto mulNode = std::make_shared<ngraph::op::v1::Multiply>(powNode, alphaNode);
    auto subNode = std::make_shared<ngraph::op::v1::Subtract>(mulNode, alphaNode);

    auto addNode = std::make_shared<ngraph::op::v1::Add>(subNode, posNode);
    addNode->validate_and_infer_types();

    addNode->set_friendly_name(paramlist->name);
    ngraph::NodeVector outputNodes;
    outputNodes.push_back(addNode);
    SetOutputNodes(outputNodes);
    
    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Selu, LAYER_SELU);

}