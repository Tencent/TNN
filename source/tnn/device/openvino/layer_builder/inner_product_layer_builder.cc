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

DECLARE_OPENVINO_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);

Status InnerProductOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<InnerProductLayerParam*>(param_);
    
    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    ngraph::Shape matShape;
    //    matShape.push_back(input_node->get_output_shape(0).at(paramlist->axis));
    matShape.push_back(paramlist->num_output);
    matShape.push_back(input_node->get_output_shape(0).at(paramlist->axis));
    size_t matSize = matShape.at(0) * matShape.at(1);

    auto resource = dynamic_cast<InnerProductLayerResource*>(GetResource());
    InferenceEngine::TBlob<float>::Ptr matPtr(new InferenceEngine::TBlob<float>({InferenceEngine::Precision::FP32, {matSize}, InferenceEngine::Layout::C}));
    matPtr->allocate();
    void* buffer = matPtr->buffer();
    float* matBuffer = reinterpret_cast<float*>(buffer);

    const float* matResource = resource->weight_handle.force_to<float*>();
    for (size_t i = 0; i < matSize; i++) matBuffer[i] = matResource[i];

    auto matNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::f32, matShape, matBuffer);
    
    // should transpose the axis to the end
    std::vector<int> transposeAxis;
    for (size_t i = 0; i < input_node->get_output_shape(0).size(); i++) {
        if (i != paramlist->axis)
            transposeAxis.push_back(i);
    }
    transposeAxis.push_back(paramlist->axis);
    ngraph::Shape transposeShape;
    transposeShape.push_back(input_node->get_output_shape(0).size());
    
    auto transposeConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, transposeShape, transposeAxis);
    
    auto transposeNode = std::make_shared<ngraph::op::Transpose>(
        input_node->output(0), transposeConst->output(0));
    
    transposeNode->validate_and_infer_types();
    
    // inner product
    auto innerProductNode = std::make_shared<ngraph::op::MatMul>(
        transposeNode->output(0), matNode->output(0), false, !paramlist->transpose);

    innerProductNode->validate_and_infer_types();

    // transpose reverse
    std::vector<int> transposeReverseAxis;
    for (size_t i = 0; i < input_node->get_output_shape(0).size() - 1; i++) {
        if (i == paramlist->axis) {
            transposeReverseAxis.push_back(input_node->get_output_shape(0).size() - 1);
        }
        transposeReverseAxis.push_back(i);
    }
    auto transposeReverseConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, transposeShape, transposeReverseAxis);
    
    auto transposeReverseNode = std::make_shared<ngraph::op::Transpose>(
        innerProductNode->output(0), transposeReverseConst->output(0));
    
    if (paramlist->has_bias) {
        auto img_size = 1;
        for (auto item : transposeReverseNode->get_output_shape(0)) {
            img_size *= item;
        }
        auto bias_size = transposeReverseNode->get_output_shape(0).at(1); // output channnel
        img_size /= bias_size;

        InferenceEngine::TBlob<float>::Ptr biasPtr(new InferenceEngine::TBlob<float>(
            {InferenceEngine::Precision::FP32, transposeReverseNode->get_output_shape(0), InferenceEngine::Layout::OIHW}));
        biasPtr->allocate();

        // extend bias shape
        void* bias_buffer = biasPtr->buffer();
        const float* w_bias = resource->bias_handle.force_to<float*>();
        for (size_t i = 0; i < bias_size; i++) {
            for (size_t j = 0; j < img_size; j++) {
                reinterpret_cast<float*>(bias_buffer)[i * img_size + j] = w_bias[i];
            }
        }

        auto biasNode = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::f32, transposeReverseNode->get_output_shape(0), biasPtr->cbuffer().as<float*>());

        // biasNode->validate_and_infer_types();
        auto addNode = std::make_shared<ngraph::op::v1::Add>(
            transposeReverseNode->output(0), biasNode->output(0));

        addNode->set_friendly_name(paramlist->name);
        addNode->validate_and_infer_types();

        ngraph::NodeVector outputNodes;
        outputNodes.push_back(addNode);
        SetOutputNodes(outputNodes);

    } else {
        transposeReverseNode->set_friendly_name(paramlist->name);
        transposeReverseNode->validate_and_infer_types();

        ngraph::NodeVector outputNodes;
        outputNodes.push_back(transposeReverseNode);
        SetOutputNodes(outputNodes);
    }
    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);
}