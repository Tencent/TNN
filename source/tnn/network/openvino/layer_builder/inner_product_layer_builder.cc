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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);

Status InnerProductOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<InnerProductLayerParam *>(param_);
    auto resource  = dynamic_cast<InnerProductLayerResource *>(resource_);
    
    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];\

    size_t m = input_node->get_output_shape(0)[0];
    size_t n = input_node->get_output_shape(0)[1] * input_node->get_output_shape(0)[2] * \
            input_node->get_output_shape(0)[3];
    size_t k = paramlist->num_output;

    std::vector<int> matShape;
    matShape.push_back(m);
    matShape.push_back(n);

    auto reshapeConstNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, ngraph::Shape({2}), matShape);
    
    auto reshapeNode = std::make_shared<ngraph::op::v1::Reshape>(
        input_node->output(0), reshapeConstNode, true);

    ngraph::Shape weightsShape;

    auto weightsNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::f32, ngraph::Shape({k, n}), resource->weight_handle.force_to<float *>());
    
    auto matMulNode  = std::make_shared<ngraph::op::MatMul>(
        reshapeNode->output(0), weightsNode->output(0), false, !paramlist->transpose);

    std::vector<int> matReshape;
    matReshape.push_back(m);
    matReshape.push_back(k);
    matReshape.push_back(1);
    matReshape.push_back(1);

    auto reverseReshapeConstNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, ngraph::Shape({4}), matReshape);
    
    auto reverseReshapeNode      = std::make_shared<ngraph::op::v1::Reshape>(
        matMulNode->output(0), reverseReshapeConstNode, true);
    
    if (paramlist->has_bias) {
        ngraph::Shape biasShape;
        auto output_shape = reverseReshapeNode->get_output_shape(0);
        for (int i = 0; i < output_shape.size(); i++) {
            if (i == paramlist->axis) {
                biasShape.push_back(output_shape.at(i));
            } else {
                biasShape.push_back(1);
            }
        }

        auto biasNode = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::f32, biasShape, resource->bias_handle.force_to<float*>());
        
        auto addNode = std::make_shared<ngraph::op::v1::Add>(
            reverseReshapeNode->output(0), biasNode->output(0));
            
        addNode->set_friendly_name(paramlist->name);
        addNode->validate_and_infer_types();

        ngraph::NodeVector outputNodes;
        outputNodes.push_back(addNode);
        SetOutputTensors(outputNodes);

    } else {
        reverseReshapeNode->set_friendly_name(paramlist->name);
        reverseReshapeNode->validate_and_infer_types();

        ngraph::NodeVector outputNodes;
        outputNodes.push_back(reverseReshapeNode);
        SetOutputTensors(outputNodes);
    }
    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);
}