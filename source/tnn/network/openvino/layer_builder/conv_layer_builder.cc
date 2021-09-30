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

class ConvOVLayerBuilder : public OpenVINOLayerBuilder {
public:
    ConvOVLayerBuilder(LayerType ignore = LAYER_CONVOLUTION) : OpenVINOLayerBuilder(ignore){};
    virtual ~ConvOVLayerBuilder(){};
protected:
    virtual Status InferOutputShape() {return TNN_OK;};
    virtual Status InferOutputDataType() {return TNN_OK;};
    virtual Status Build();
};

class Conv1DOVLayerBuilder : public ConvOVLayerBuilder {
public:
    Conv1DOVLayerBuilder(LayerType ignore = LAYER_CONVOLUTION_1D) : ConvOVLayerBuilder(ignore){};
};

Status ConvOVLayerBuilder::Build() {
    auto paramlist = dynamic_cast<ConvLayerParam*>(param_);

    if (GetInputNodes().size() <= 0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    auto convNode = std::make_shared<ngraph::op::v1::GroupConvolution>();

    // set strides
    ngraph::Strides stride;
    for (auto item : paramlist->strides) {
        stride.push_back(item);
    }
    std::reverse(stride.begin(), stride.end());
    convNode->set_strides(stride);

    // set pads
    ngraph::CoordinateDiff pad_begin, pad_end;
    if (paramlist->pads.size() == 2) {
        pad_begin.push_back(paramlist->pads.at(0));
        pad_end.push_back(paramlist->pads.at(1));
    } else if (paramlist->pads.size() == 4) {
        pad_begin.push_back(paramlist->pads.at(2));
        pad_begin.push_back(paramlist->pads.at(0));
        pad_end.push_back(paramlist->pads.at(3));
        pad_end.push_back(paramlist->pads.at(1));
    }
    convNode->set_pads_begin(pad_begin);
    convNode->set_adding_above(pad_end);

    // set dilations
    ngraph::Strides dilation;
    for (auto item : paramlist->dialations) {
        dilation.push_back(item);
    }
    std::reverse(dilation.begin(), dilation.end());
    convNode->set_dilations(dilation);

    // set pad type
    ngraph::op::PadType pad_type;
    if (paramlist->pad_type == -1) {
        pad_type = ngraph::op::PadType::EXPLICIT;
    } else if (paramlist->pad_type == 0) {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    } else {
        pad_type = ngraph::op::PadType::VALID;
    }
    convNode->set_auto_pad(pad_type);

    // set weights
    size_t weight_size = 1;
    ngraph::Shape weights_shape;
    weights_shape.push_back(paramlist->group);
    weights_shape.push_back(paramlist->output_channel / paramlist->group);
    weights_shape.push_back(input_node->get_output_shape(0).at(1) / paramlist->group);
    weight_size *= paramlist->output_channel * paramlist->input_channel;
    for (int i = paramlist->kernels.size() - 1; i >= 0; i--) {
        weights_shape.push_back(paramlist->kernels.at(i));
        weight_size *= paramlist->kernels.at(i);
    }

    auto resource = dynamic_cast<ConvLayerResource*>(GetResource());

    std::shared_ptr<ngraph::Node> weights_Node = std::make_shared<ngraph::op::Constant>(
        DataTransfer(resource->filter_handle.GetDataType()), weights_shape, resource->filter_handle.force_to<float*>());

    // if input channels > weights input channels
    if (input_node->get_output_shape(0).at(1) > paramlist->input_channel * paramlist->group) {
        auto channels = paramlist->input_channel * paramlist->group;
        ngraph::Shape axisShape, lengthShape;
        axisShape.push_back(1);
        lengthShape.push_back(2);
        auto axisNode =
            std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i32, axisShape, std::vector<int>({1}));
        auto lengthNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i32, lengthShape,
                                                                 std::vector<int>({channels, -1}));
        auto sliceNode  = std::make_shared<ngraph::op::VariadicSplit>(input_node->output(0), axisNode, lengthNode);
        convNode->set_argument(0, sliceNode->output(0));
    } else {
        convNode->set_argument(0, input_node->output(0));
    }
    convNode->set_argument(1, weights_Node->output(0));
    convNode->validate_and_infer_types();

    std::shared_ptr<ngraph::Node> output_node = nullptr;
    ngraph::NodeVector outputNodes;

    if (paramlist->bias) {
        // set bias shape
        ngraph::Shape biasShape;
        for (size_t i = 0; i < convNode->get_output_shape(0).size(); i++) {
            if (i == 1)
                biasShape.push_back(convNode->get_output_shape(0).at(1));
            else
                biasShape.push_back(1);
        }

        // set bias node
        std::shared_ptr<ngraph::Node> biasNode = std::make_shared<ngraph::op::Constant>(
            DataTransfer(resource->bias_handle.GetDataType()), biasShape, resource->bias_handle.force_to<float*>());

        auto addNode = std::make_shared<ngraph::op::v1::Add>();
        addNode->set_argument(0, convNode->output(0));
        addNode->set_argument(1, biasNode->output(0));
        addNode->validate_and_infer_types();
        output_node = addNode;
    } else {
        output_node = convNode;
    }

    if (paramlist->activation_type != ActivationType_None) {
        if (paramlist->activation_type == ActivationType_ReLU) {
            output_node = std::make_shared<ngraph::op::Relu>(output_node->output(0));
        } else if (paramlist->activation_type == ActivationType_ReLU6) {
            output_node = std::make_shared<ngraph::op::Clamp>(output_node->output(0), 0, 6);
        } else {
            return Status(TNNERR_PARAM_ERR, "Unsupported activation type");
        }
    }

    output_node->validate_and_infer_types();
    output_node->set_friendly_name(paramlist->name);
    outputNodes.push_back(output_node);
    SetOutputTensors(outputNodes);
    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Conv, LAYER_CONVOLUTION);
REGISTER_OPENVINO_LAYER_BUILDER(Conv1D, LAYER_CONVOLUTION_1D);

}  // namespace TNN_NS
