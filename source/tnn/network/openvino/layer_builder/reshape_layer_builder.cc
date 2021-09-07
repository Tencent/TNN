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

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/layer/base_layer.h"
#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/network/openvino/openvino_types.h"
#include "tnn/network/openvino/custom_layer/custom_reshape.h"
#include "tnn/network/openvino/utils.h"

namespace TNN_NS {
DECLARE_OPENVINO_LAYER_BUILDER(Reshape, LAYER_RESHAPE);

Status ReshapeOVLayerBuilder::Build() {
    auto paramlist = dynamic_cast<ReshapeLayerParam*>(param_);

    if (GetInputNodes().size() <= 0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    ngraph::Shape output_shape;
    output_shape.push_back(paramlist->num_axes);

    std::vector<int> shapePattern;
    for (auto item : paramlist->shape) {
        shapePattern.push_back(item);
    }
    auto patternNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i32, output_shape, shapePattern);

    std::shared_ptr<ngraph::Node> reshapeNode = nullptr;
    if (paramlist->reshape_type != 1 && input_blobs_.size() == 1) {
        reshapeNode = std::make_shared<ngraph::op::v1::Reshape>(input_node->output(0), patternNode, true);
        reshapeNode->set_friendly_name(paramlist->name);
        reshapeNode->validate_and_infer_types();
        ngraph::NodeVector outputNodes;
        outputNodes.push_back(reshapeNode);
        SetOutputTensors(outputNodes);
    } else {
        ADD_CUSTOM_NODE(Reshape, paramlist->name);
    }

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Reshape, LAYER_RESHAPE);
REGISTER_CUSTOM_TYPE(LAYER_RESHAPE);
}  // namespace TNN_NS