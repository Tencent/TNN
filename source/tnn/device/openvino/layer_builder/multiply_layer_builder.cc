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

DECLARE_OPENVINO_LAYER_BUILDER(Mul, LAYER_MUL);

Status MulOVLayerBuilder::Build() {

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes();

    std::shared_ptr<ngraph::op::v1::Multiply> mulNode;
    if (input_node.size() == 2) {
        mulNode = std::make_shared<ngraph::op::v1::Multiply>(
            input_node[0]->output(0), input_node[1]->output(0));
    } else {
        auto resource = dynamic_cast<EltwiseLayerResource*>(GetResource());
        // suppose that weights are on channels broadcast
        ngraph::Shape weightNodeShape;
        for (size_t i = 0; i < input_node[0]->get_output_shape(0).size(); i++) {
            if (i == 1) weightNodeShape.push_back(resource->element_handle.GetDataCount());
            else weightNodeShape.push_back(1);
        }

        auto weightNode = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::f32, weightNodeShape, resource->element_handle.force_to<float*>());
        weightNode->validate_and_infer_types();

        mulNode = std::make_shared<ngraph::op::v1::Multiply>(
            input_node[0]->output(0), weightNode);
    }
    mulNode->validate_and_infer_types();
    mulNode->set_friendly_name(param_->name);

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(mulNode);
    SetOutputNodes(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Mul, LAYER_MUL);

}