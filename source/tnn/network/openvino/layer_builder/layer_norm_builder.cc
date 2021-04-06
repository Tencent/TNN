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
#include "tnn/network/openvino/custom_layer/custom_layer_norm.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(LayerNorm, LAYER_LAYER_NORM);

Status LayerNormOVLayerBuilder::Build() {
    
    auto paramlist = dynamic_cast<LayerNormLayerParam*>(param_);

    if (GetInputNodes().size() <= 2) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }

    auto input_nodes = GetInputNodes();

    auto input_shape = input_nodes[0]->get_output_shape(0);
    std::vector<int> NormAxes;
    for (int i = paramlist->reduce_dims_size; i < input_shape.size(); i++) {
        NormAxes.push_back(i); 
    }

    if (0) {
        auto AxesNode = std::make_shared<ngraph::op::Constant>(ngraph::element::i32, ngraph::Shape{NormAxes.size()}, NormAxes);
        auto MVNNode = std::make_shared<ngraph::op::v6::MVN>(input_nodes[0]->output(0), AxesNode->output(0), true, 1e-05f, ngraph::op::MVNEpsMode::INSIDE_SQRT);
        MVNNode->validate_and_infer_types();
        auto scaleNode = std::make_shared<ngraph::op::v1::Multiply>(MVNNode->output(0), input_nodes[1]->output(0));
        scaleNode->validate_and_infer_types();
        auto biasNode  = std::make_shared<ngraph::op::v1::Add>(scaleNode->output(0), input_nodes[2]->output(0));
        biasNode->validate_and_infer_types();

        biasNode->set_friendly_name(param_->name);
        ngraph::NodeVector outputNodes;
        outputNodes.push_back(biasNode);
        SetOutputTensors(outputNodes);
    } else {
        ngraph::OutputVector inputs;
        for (auto item : input_nodes) {
            inputs.push_back(item->output(0));
        }        
        auto instNormNode = std::make_shared<CustomLayerNormOp>(inputs, base_layer_, GetInputBlobs(), GetOutputBlobs());
        instNormNode->set_friendly_name(param_->name);
        ngraph::NodeVector outputNodes;
        outputNodes.push_back(instNormNode);
        SetOutputTensors(outputNodes);
    }


    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(LayerNorm, LAYER_LAYER_NORM);
REGISTER_CUSTOM_TYPE(LAYER_LAYER_NORM);

}