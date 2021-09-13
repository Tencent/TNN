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
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(Scale, LAYER_SCALE);

Status ScaleOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<ScaleLayerParam*>(param_);

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    auto resource = dynamic_cast<BatchNormLayerResource*>(GetResource());
    bool share_channel = resource->scale_handle.GetBytesSize() == DataTypeUtils::GetBytesSize(resource->scale_handle.GetDataType());
    auto *scale_data = resource->scale_handle.force_to<float*>();
    auto *bias_data = resource->bias_handle.force_to<float*>();

    auto input_shape = input_node->get_output_shape(0);
    ngraph::Shape batchNromShape;
    for (size_t i = 0; i < input_shape.size(); i++) {
        if (i == 1 && !share_channel) batchNromShape.push_back(input_shape.at(1));
        else batchNromShape.push_back(1); 
    }
    auto scaleConstNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::f32, batchNromShape, scale_data);
    auto scaleNode = std::make_shared<ngraph::op::v1::Multiply>(
        input_node->output(0), scaleConstNode);

    bool has_bias = resource->bias_handle.GetBytesSize() > 0;
    if (has_bias) {
        auto biasConstNode = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::f32, batchNromShape, bias_data);
        auto biasNode = std::make_shared<ngraph::op::v1::Add>(
            scaleNode->output(0), biasConstNode, ngraph::op::AutoBroadcastType::NUMPY);
        biasNode->set_friendly_name(param_->name);
        ngraph::NodeVector outputNodes = {biasNode};
        SetOutputTensors(outputNodes);
    } else {
        scaleNode->set_friendly_name(param_->name);
        ngraph::NodeVector outputNodes = {scaleNode};
        SetOutputTensors(outputNodes);
    }
    
    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Scale, LAYER_SCALE);

}