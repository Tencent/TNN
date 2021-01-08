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

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(Permute, LAYER_PERMUTE);

Status PermuteOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<PermuteLayerParam*>(param_);

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    size_t size = paramlist->orders.size();

    // set transpose node
    ngraph::Shape permuteShape;
    permuteShape.push_back(size);
    std::vector<int> permutePattern;
    for (auto item : paramlist->orders) {
        permutePattern.push_back(item);
    }

    auto orderNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, permuteShape, permutePattern);
    
    // set transpose Node
    auto permuteNode = std::make_shared<ngraph::op::Transpose>(
        input_node->output(0), orderNode);
    permuteNode->set_friendly_name(paramlist->name);
    permuteNode->validate_and_infer_types();

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(permuteNode);
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Permute, LAYER_PERMUTE);

}