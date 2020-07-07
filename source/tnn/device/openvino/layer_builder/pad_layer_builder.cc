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

DECLARE_OPENVINO_LAYER_BUILDER(Pad, LAYER_PAD);

Status PadOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<PadLayerParam*>(param_);

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    // set pad node
    std::vector<int> beginPattern, endPattern;
    for (size_t i = 0; i < 2; i++) {
        beginPattern.push_back(0);
        endPattern.push_back(0);
    }

    beginPattern.push_back(paramlist->pads.at(2));
    beginPattern.push_back(paramlist->pads.at(0));
    endPattern.push_back(paramlist->pads.at(3));
    endPattern.push_back(paramlist->pads.at(1));

    auto pad_begin = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, ngraph::Shape{4}, beginPattern);
    auto pad_end = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, ngraph::Shape{4}, endPattern);

    ngraph::op::PadMode padMode;
    if (paramlist->type == 0) {
        padMode = ngraph::op::PadMode::CONSTANT;
    } else if (paramlist->type == 1) {
        padMode = ngraph::op::PadMode::REFLECT;
    } else {
        padMode = ngraph::op::PadMode::EDGE;
    }

    auto padNode = std::make_shared<ngraph::op::v1::Pad>(
        input_node->output(0), pad_begin, pad_end, padMode);

    padNode->validate_and_infer_types();
    padNode->set_friendly_name(paramlist->name);

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(padNode);
    SetOutputNodes(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Pad, LAYER_PAD);

}