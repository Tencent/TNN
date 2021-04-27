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

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(StrideSlice, LAYER_STRIDED_SLICE);

Status StrideSliceOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<StrideSliceLayerParam*>(param_);
    
    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    ngraph::Shape strideSliceShape;
    strideSliceShape.push_back(paramlist->begins.size());

    std::vector<int> begins, ends, strides;
    std::vector<int64_t> begin_mask, end_mask;
    for (int i = paramlist->begins.size() - 1; i > -1; i--) {
        if (paramlist->begins.at(i) == 0) begin_mask.push_back(1);
        else begin_mask.push_back(0);
        if (paramlist->ends.at(i) == 0) end_mask.push_back(1);
        else end_mask.push_back(0);
        begins.push_back(paramlist->begins.at(i));
        ends.push_back(paramlist->ends.at(i));
        strides.push_back(paramlist->strides.at(i));
    }

    auto beginNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, strideSliceShape, begins);
    auto endNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, strideSliceShape, ends);
    auto strideNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, strideSliceShape, strides);

    auto strideSliceNode = std::make_shared<ngraph::op::v1::StridedSlice>(
        input_node->output(0), beginNode, endNode, strideNode, begin_mask, end_mask);
    
    strideSliceNode->validate_and_infer_types();
    strideSliceNode->set_friendly_name(paramlist->name);

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(strideSliceNode);
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(StrideSlice, LAYER_STRIDED_SLICE);
}