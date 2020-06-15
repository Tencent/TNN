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

DECLARE_OPENVINO_LAYER_BUILDER(Pooling, LAYER_POOLING);

Status PoolingOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<PoolingLayerParam*>(param_);
    
    if (GetInputNodes().size() <= 0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes();

    // set strides
    ngraph::Strides strides;
    for (auto item : paramlist->strides) {
        strides.push_back(item);
    }
    std::reverse(strides.begin(), strides.end());

    // set pads
    ngraph::Shape pad_begin, pad_end;
    pad_begin.push_back(paramlist->pads.at(2));
    pad_begin.push_back(paramlist->pads.at(0));
    pad_end.push_back(paramlist->pads.at(3));
    pad_end.push_back(paramlist->pads.at(1));

    // set rounding
    ngraph::op::RoundingType rounding_type;
    if (paramlist->ceil_mode == 1) {
        rounding_type = ngraph::op::RoundingType::CEIL;
    } else {
        rounding_type = ngraph::op::RoundingType::FLOOR;
    }

    // set pad type
    ngraph::op::PadType pad_type ; //= ngraph::op::PadType::EXPLICIT;
    if (paramlist->pad_type == -1) {
        pad_type = ngraph::op::PadType::EXPLICIT;
    } else if (paramlist->pad_type == 0) {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    } else {
        pad_type = ngraph::op::PadType::VALID;
    }
    
    // kernel shape
    std::reverse(paramlist->kernels.begin(), paramlist->kernels.end());
    ngraph::Shape kernel_shape;
    for (size_t i = 0; i < 2; i++) {
        if (paramlist->kernels.at(i) == 0) {
            kernel_shape.push_back(input_node[0]->output(0).get_shape().at(i+2));
        } else {
            kernel_shape.push_back(paramlist->kernels.at(i));
        }
    }

    std::shared_ptr<ngraph::Node> poolNode;
    if (paramlist->pool_type == 0) { // max pool
        poolNode = std::make_shared<ngraph::op::v1::MaxPool>(
            input_node[0]->output(0), strides, pad_begin, pad_end, kernel_shape, rounding_type, pad_type);
        poolNode->validate_and_infer_types();
    } else {    // average pool
        poolNode = std::make_shared<ngraph::op::v1::AvgPool>(
            input_node[0]->output(0), strides, pad_begin, pad_end, kernel_shape, true, rounding_type, pad_type);        
        poolNode->validate_and_infer_types();
    }

    poolNode->set_friendly_name(paramlist->name);

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(poolNode);
    SetOutputNodes(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Pooling, LAYER_POOLING);

}