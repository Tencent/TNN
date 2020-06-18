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
#include <inference_engine.hpp>

#include "tnn/layer/base_layer.h"
#include "tnn/device/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/device/openvino/openvino_types.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(Conv, LAYER_CONVOLUTION);

Status ConvOVLayerBuilder::Build() {
    
    auto paramlist = dynamic_cast<ConvLayerParam*>(this->param_);
    
    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto in_node = GetInputNodes()[0];
    
    // std::cout << "building conv node" << std::endl;
    auto convNode = std::make_shared<ngraph::op::v1::Convolution>();

    // paramlist->strides;
    // paramlist->pads;
    // paramlist->kernels;
    // paramlist->input_channel;
    // paramlist->output_channel;
    // paramlist->name;
    // paramlist->pad_type;
    // paramlist->weight_data_size;
    // paramlist->bias;
    // paramlist->dialations;

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

    convNode->set_auto_pad(ngraph::op::PadType::EXPLICIT); // 这里需要有一定的对应 pad_type -> PadType

    // set weights
    convNode->set_argument(0, inputNodes_.at(0)->output(0));
    ngraph::Shape weights_shape;
    weights_shape.push_back(paramlist->output_channel);
    weights_shape.push_back(paramlist->input_channel);
    for (auto item : paramlist->kernels) {
        weights_shape.push_back(item);
    }
    auto resource = dynamic_cast<ConvLayerResource*>(GetResource());
    // resource->filter_handle.force_to<ngraph::element::f32>();
    // resource->filter_handle.SetDataType(tnn::DataType::DATA_TYPE_FLOAT);
    const float *w_scale = resource->scale_handle.force_to<float *>();
    InferenceEngine::TBlob<float>::Ptr weightsPtr(new InferenceEngine::TBlob<float>({InferenceEngine::Precision::FP32, {paramlist->weight_data_size}, InferenceEngine::Layout::C}));
    weightsPtr->allocate();
    void* buffer = weightsPtr->buffer();
    auto weight_buffer = reinterpret_cast<float*>(buffer);
    for (size_t i = 0; i < paramlist->weight_data_size; i++) {
        weight_buffer[i] = w_scale[i];
    }
    // std::cout << "setting weight buffer" << std::endl;
    std::shared_ptr<ngraph::Node> weights_Node = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::f32, weights_shape, weightsPtr->cbuffer().as<float*>());
    convNode->set_argument(1, weights_Node);

    // set node name
    convNode->set_friendly_name(paramlist->name);

    // set output node
    ngraph::NodeVector outputNodes;
    outputNodes.push_back(convNode);
    convNode->validate_and_infer_types();
    SetOutputNodes(outputNodes);

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
