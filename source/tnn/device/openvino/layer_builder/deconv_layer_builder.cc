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

DECLARE_OPENVINO_LAYER_BUILDER(Deconv, LAYER_DECONVOLUTION);

Status DeconvOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<ConvLayerParam*>(param_);

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    // set strides
    ngraph::Strides strides;
    for (auto item : paramlist->strides) {
        strides.push_back(item);
    }
    std::reverse(strides.begin(), strides.end());

    // set pads
    ngraph::CoordinateDiff pads_begin, pads_end;
    pads_begin.push_back(paramlist->pads.at(2));
    pads_begin.push_back(paramlist->pads.at(0));
    pads_end.push_back(paramlist->pads.at(3));
    pads_end.push_back(paramlist->pads.at(1));

    // set dilations
    ngraph::Strides dilations;
    for (auto item : paramlist->dialations) {
        dilations.push_back(item);
    }
    std::reverse(dilations.begin(), dilations.end());

    // set pad type
    ngraph::op::PadType pad_type;
    if (paramlist->pad_type == -1) {
        pad_type = ngraph::op::PadType::EXPLICIT;
    } else if (paramlist->pad_type == 0) {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    } else {
        pad_type = ngraph::op::PadType::VALID;
    }

    // set weights
    ngraph::Shape weights_shape;
    weights_shape.push_back(paramlist->group);
    weights_shape.push_back(paramlist->output_channel / paramlist->group);
    weights_shape.push_back(paramlist->input_channel);
    auto kernels = std::vector<int>{paramlist->kernels};
    std::reverse(kernels.begin(), kernels.end());
    for (auto item : kernels) {
        weights_shape.push_back(item);
    }
    
    auto resource = dynamic_cast<ConvLayerResource*>(GetResource());
    auto weightsNode = std::make_shared<ngraph::op::Constant>(
        DataTransfer(resource->filter_handle.GetDataType()), weights_shape, resource->filter_handle.force_to<float*>());

    // assume that channels == weights input channels
    auto deConvNode = std::make_shared<ngraph::op::v1::GroupConvolutionBackpropData>(
        input_node->output(0), weightsNode, strides, pads_begin, pads_end, dilations, pad_type);
    deConvNode->validate_and_infer_types();

    // has bias
    if (paramlist->bias) {
        // set bias shape
        ngraph::Shape biasShape;
        for (size_t i = 0; i < deConvNode->get_output_shape(0).size(); i++) {
            if (i == 1) biasShape.push_back(resource->bias_handle.GetDataCount());
            else biasShape.push_back(1);
        }

        // set bias node 
        auto biasNode = std::make_shared<ngraph::op::Constant>(
            DataTransfer(resource->bias_handle.GetDataType()), biasShape, resource->bias_handle.force_to<float*>());
        auto addNode = std::make_shared<ngraph::op::v1::Add>(
            deConvNode->output(0), biasNode);
        addNode->validate_and_infer_types();

        addNode->set_friendly_name(paramlist->name);
        ngraph::NodeVector outputNodes;
        outputNodes.push_back(addNode);
        SetOutputNodes(outputNodes);

    } else {
        deConvNode->set_friendly_name(paramlist->name);
        ngraph::NodeVector outputNodes;
        outputNodes.push_back(deConvNode);
        SetOutputNodes(outputNodes);
    }

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Deconv, LAYER_DECONVOLUTION);

}