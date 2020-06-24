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

DECLARE_OPENVINO_LAYER_BUILDER(Conv, LAYER_CONVOLUTION);

Status ConvOVLayerBuilder::Build() {
    
    auto paramlist = dynamic_cast<ConvLayerParam*>(param_);
    
    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto in_node = GetInputNodes()[0];

    auto convNode = std::make_shared<ngraph::op::v1::Convolution>();
    convNode->set_argument(0, in_node->output(0));


    // set strides
    ngraph::Strides stride;
    for (auto item : paramlist->strides) {
        stride.push_back(item);
    }
    convNode->set_strides(stride);

    // set pads
    ngraph::CoordinateDiff pad_begin, pad_end;
    pad_begin.push_back(paramlist->pads.at(0));
    pad_begin.push_back(paramlist->pads.at(2));
    pad_end.push_back(paramlist->pads.at(1));
    pad_end.push_back(paramlist->pads.at(3));
    convNode->set_pads_begin(pad_begin);
    convNode->set_adding_above(pad_end);

    // set dilations
    ngraph::Strides dilation;
    for (auto item : paramlist->dialations) {
        dilation.push_back(item);
    }
    convNode->set_dilations(dilation);

    // set pad type
    ngraph::op::PadType pad_type;
    if (paramlist->pad_type == -1) {
        pad_type = ngraph::op::PadType::EXPLICIT;
    } else if (paramlist->pad_type == 0) {
        pad_type = ngraph::op::PadType::SAME_LOWER;
    } else {
        pad_type = ngraph::op::PadType::VALID;
    }
    convNode->set_auto_pad(pad_type);

    // set weights
    size_t weight_size = 1;
    ngraph::Shape weights_shape;
    weights_shape.push_back(paramlist->output_channel);
    weights_shape.push_back(paramlist->input_channel);
    weight_size *= paramlist->output_channel * paramlist->input_channel;
    for (auto item : paramlist->kernels) {
        weights_shape.push_back(item);
        weight_size *= item;
    }
    InferenceEngine::TBlob<float>::Ptr weightsPtr(new InferenceEngine::TBlob<float>({InferenceEngine::Precision::FP32, weights_shape, InferenceEngine::Layout::OIHW}));
    weightsPtr->allocate();
    void* buffer = weightsPtr->buffer();
    auto weight_buffer = reinterpret_cast<float*>(buffer);

    auto resource = dynamic_cast<ConvLayerResource*>(GetResource());
    const float *w_scale = resource->filter_handle.force_to<float *>();

    for (size_t i = 0; i < weight_size; i++) {
        weight_buffer[i] = w_scale[i];
    }

    std::shared_ptr<ngraph::Node> weights_Node = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::f32, weights_shape, weightsPtr->cbuffer().as<float*>());
    convNode->set_argument(1, weights_Node->output(0));
    convNode->validate_and_infer_types();

    if (paramlist->bias) {

        // set bias
        auto img_size = 1;
        for (auto item : convNode->get_output_shape(0)) {
            img_size *= item;
        }
        auto bias_size = paramlist->output_channel;
        img_size /= bias_size;

        InferenceEngine::TBlob<float>::Ptr biasPtr(new InferenceEngine::TBlob<float>({InferenceEngine::Precision::FP32, convNode->get_output_shape(0), InferenceEngine::Layout::OIHW}));
        biasPtr->allocate();

        // extend bias to conv shape
        void* bias_buffer = biasPtr->buffer();
        const float* w_bias = resource->bias_handle.force_to<float*>();
        for (size_t i = 0; i < bias_size; i++) {
            for (size_t j =0; j < img_size; j++) {
                reinterpret_cast<float*>(bias_buffer)[i * img_size + j] = w_bias[i];
            }
        }
        
        std::shared_ptr<ngraph::Node> biasNode = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::f32, convNode->get_output_shape(0), biasPtr->cbuffer().as<float*>());
        
        auto addNode = std::make_shared<ngraph::op::v1::Add>();
        addNode->set_argument(0, convNode->output(0));
        addNode->set_argument(1, biasNode->output(0));

        addNode->set_friendly_name(paramlist->name);
        ngraph::NodeVector outputNodes;
        outputNodes.push_back(addNode);
        addNode->validate_and_infer_types();
        SetOutputNodes(outputNodes);
        // std::dynamic_pointer_cast<OpenvinoTensor>(GetOutputTensors()[0])->SetNode(addNode);
    } else {
        // set node name
        convNode->set_friendly_name(paramlist->name);

        // set output node
        ngraph::NodeVector outputNodes;
        outputNodes.push_back(convNode);
        convNode->validate_and_infer_types();
        SetOutputNodes(outputNodes);
        // std::dynamic_pointer_cast<OpenvinoTensor>(GetOutputTensors()[0])->SetNode(convNode);
    }
    // build the conv layer and generates a new out_node. 

    // here simply asign out_node = in_node for code compiling.
    // auto out_node = in_node;

    // if (GetOutputTensors().size() <=0) {
    //     LOGE("Error: 0 output tensors\n");
    //     return TNNERR_INIT_LAYER;
    // }

    // std::dynamic_pointer_cast<OpenvinoTensor>(GetOutputTensors()[0])->SetNode(out_node);
    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Conv, LAYER_CONVOLUTION);

}  // namespace TNN_NS
