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

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(Shape, LAYER_SHAPE);

Status ShapeOVLayerBuilder::Build() {
    auto paramlist = param_;

    if (GetInputNodes().size() <= 0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node  = GetInputNodes()[0];
    auto input_shape = input_node->get_output_shape(0);

    auto shapeNode = std::make_shared<ngraph::op::ShapeOf>(input_node->output(0));

    shapeNode->validate_and_infer_types();
    shapeNode->set_friendly_name(paramlist->name);

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(shapeNode);
    SetOutputTensors(outputNodes);

    return TNN_OK;

    // auto input_node = GetInputNodes();

    // // auto input_blobs = GetInputBlobs();
    // // auto output_blobs = GetOutputBlobs();
    // // print(input_blobs[0]->GetBlobDesc().dims);
    // // print(output_blobs[0]->GetBlobDesc().dims);

    // ngraph::OutputVector inputs;
    // for (auto item : input_node) {
    //     inputs.push_back(item->output(0));
    // }
    // auto unsqueezeNode = std::make_shared<CustomShapeOp>(
    //     inputs, base_layer_, GetInputBlobs(), GetOutputBlobs());

    // unsqueezeNode->validate_and_infer_types();
    // unsqueezeNode->set_friendly_name(param_->name);

    // ngraph::NodeVector outputNodes;
    // outputNodes.push_back(unsqueezeNode);
    // SetOutputTensors(outputNodes);

    // return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Shape, LAYER_SHAPE);

}  // namespace TNN_NS