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

DECLARE_OPENVINO_LAYER_BUILDER(Normalize, LAYER_NORMALIZE);

Status NormalizeOVLayerBuilder::Build() {
    
    auto paramlist = dynamic_cast<NormalizeLayerParam*>(param_);

    int p = paramlist->p;
    if ((p != 1 && p != 2 && p != INT_MAX && p != INT_MIN) || paramlist->axis != 1 || paramlist->across_spatial != 0) {
        LOGE("Error: Normalize layer param is not supported now\n");
        return Status(TNNERR_INST_ERR, "Error: Normalize layer param is not supported now");
    }

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    // channle shared, across_spatial not use
    size_t axisNum = 0;
    std::vector<int> axis;
    for (size_t i = 2; i < input_node->get_output_shape(0).size(); i++) { // default norm image
        axisNum++;
        axis.push_back(i);
    }
    auto axisNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, ngraph::Shape{axisNum}, axisNum);
    
    auto epsilonNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::f32, ngraph::Shape(input_node->get_output_shape(0).size(), 1), \
        std::vector<float>{paramlist->epsilon});

    std::shared_ptr<ngraph::Node> reduceNode;
    if (p == 1) {
        auto absNode = std::make_shared<ngraph::op::Abs>(input_node->output(0));
        reduceNode = std::make_shared<ngraph::op::v1::ReduceSum>(
            absNode, axisNode, true);
    } else if (p == 2) {
        auto squareNode = std::make_shared<ngraph::op::v1::Multiply>(
            input_node->output(0), input_node->output(0));
        reduceNode = std::make_shared<ngraph::op::v1::ReduceSum>(
            squareNode, axisNode, true);
        reduceNode = std::make_shared<ngraph::op::v1::Maximum>(
            squareNode, epsilonNode);
    } else if (p == INT_MAX) {
        reduceNode = std::make_shared<ngraph::op::v1::ReduceMax>(
            input_node->output(0), axisNode, true);
    } else if (p == INT_MIN) {
        reduceNode = std::make_shared<ngraph::op::v1::ReduceMin>(
            input_node->output(0), axisNode, true);
    }

    auto divNode = std::make_shared<ngraph::op::v1::Divide>(
        input_node->output(0), reduceNode);
    divNode->validate_and_infer_types();

    divNode->set_friendly_name(paramlist->name);
    ngraph::NodeVector outputNodes;
    outputNodes.push_back(divNode);
    SetOutputNodes(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Normalize, LAYER_NORMALIZE);

}