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

DECLARE_OPENVINO_LAYER_BUILDER(ShuffleChannel, LAYER_SHUFFLE_CHANNEL);

Status ShuffleChannelOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<ShuffleLayerParam*>(param_);
    
    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    // reshape-transpose-reshape
    std::vector<int> reshapePattern;
    auto input_shape = input_node->get_output_shape(0);

    reshapePattern.push_back(input_shape.at(0));
    reshapePattern.push_back(paramlist->group);
    reshapePattern.push_back(input_shape.at(1) / paramlist->group);
    for (size_t i = 2; i < input_shape.size(); i++) {
        reshapePattern.push_back(input_shape.at(i));
    }

    ngraph::Shape shuffleShape;
    shuffleShape.push_back(input_shape.size() + 1);
    
    auto patternNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, shuffleShape, reshapePattern);
    
    auto reshapeNode = std::make_shared<ngraph::op::v1::Reshape>(
        input_node->output(0), patternNode, true);
    
    reshapeNode->validate_and_infer_types();
    
    // transpose
    std::vector<int> transposeAxis;
    for (size_t i = 0; i <= input_shape.size(); i++) {
        if (i == 1) {
            transposeAxis.push_back(2);
        } else if (i == 2) {
            transposeAxis.push_back(1);
        } else {
            transposeAxis.push_back(i);
        }
    }
    ngraph::Shape transposeShape;
    transposeShape.push_back(input_shape.size() + 1);
    
    auto transposeConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, transposeShape, transposeAxis);
    
    auto transposeNode = std::make_shared<ngraph::op::Transpose>(
        reshapeNode->output(0), transposeConst->output(0));
    
    transposeNode->validate_and_infer_types();
    
    // reshape
    ngraph::Shape shuffleShape1;
    shuffleShape1.push_back(input_shape.size());
    std::vector<int> reshapePattern1;
    for (auto item : input_shape) {
        reshapePattern1.push_back(item);
    }
    
    auto patternNode1 = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, shuffleShape1, reshapePattern1);
    
    auto reshapeNode1 = std::make_shared<ngraph::op::v1::Reshape>(
        transposeNode->output(0), patternNode1->output(0), true);

    reshapeNode1->set_friendly_name(paramlist->name);
    reshapeNode1->validate_and_infer_types();

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(reshapeNode1);
    SetOutputNodes(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(ShuffleChannel, LAYER_SHUFFLE_CHANNEL);

}
