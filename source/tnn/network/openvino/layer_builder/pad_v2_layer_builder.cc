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
#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/network/openvino/openvino_types.h"

#include "tnn/network/openvino/custom_layer/custom_pad_v2.h"
#include "tnn/network/openvino/utils.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(PadV2, LAYER_PADV2);

Status PadV2OVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<PadLayerParam*>(param_);

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }

    if (paramlist->pads.size() % 2 != 0) {
        return Status(TNNERR_PARAM_ERR, "Error: Pads size must be even\n");
    }

    if (paramlist->type != 0) {
        return Status(TNNERR_PARAM_ERR, "Error: padv2 layer param is not supported");
    }

    // // set pad node
    // std::vector<int> beginPattern, endPattern;
    // for (int i = paramlist->pads.size() / 2 - 1; i >= 0; i--) {
    //     beginPattern.push_back(paramlist->pads[i*2]);
    //     endPattern.push_back(paramlist->pads[i*2+1]);
    // }

    // auto pad_begin = std::make_shared<ngraph::op::Constant>(
    //     ngraph::element::Type_t::i32, ngraph::Shape{paramlist->pads.size()/2}, beginPattern);
    // auto pad_end = std::make_shared<ngraph::op::Constant>(
    //     ngraph::element::Type_t::i32, ngraph::Shape{paramlist->pads.size()/2}, endPattern);
    // auto pad_value = std::make_shared<ngraph::op::Constant>(
    //     ngraph::element::Type_t::f32, ngraph::Shape{}, paramlist->value);

    // ngraph::op::PadMode padMode;
    // if (paramlist->type == 0) {
    //     padMode = ngraph::op::PadMode::CONSTANT;
    // } else if (paramlist->type == 1) {
    //     padMode = ngraph::op::PadMode::REFLECT;
    // } else {
    //     padMode = ngraph::op::PadMode::EDGE;
    // }

    // auto padNode = std::make_shared<ngraph::op::v1::Pad>(
    //     input_node[0]->output(0), pad_begin, pad_end, pad_value, padMode);

    // padNode->validate_and_infer_types();
    // padNode->set_friendly_name(paramlist->name);

    // ngraph::NodeVector outputNodes;
    // outputNodes.push_back(padNode);
    // SetOutputTensors(outputNodes);


    ADD_CUSTOM_NODE(PadV2, paramlist->name);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(PadV2, LAYER_PADV2);
REGISTER_CUSTOM_TYPE(LAYER_PADV2);
}