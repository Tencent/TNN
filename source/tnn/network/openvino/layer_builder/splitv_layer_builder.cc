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

DECLARE_OPENVINO_LAYER_BUILDER(Splitv, LAYER_SPLITV);

Status SplitvOVLayerBuilder::Build() {
    
    if (GetInputNodes().size() <= 0) {
        LOGE("Error: %d input nodes\n", GetInputNodes().size());
        return TNNERR_INIT_LAYER;
    }
    auto param = dynamic_cast<SplitVLayerParam *>(param_);

    auto input_node = GetInputNodes()[0];
    std::vector<int> begins, ends;
    std::vector<int64_t> begin_mask, end_mask;
    size_t input_dims = input_node->get_input_shape(0).size();
    ngraph::Shape dims_shape({input_dims});

    for (int i = 0; i < input_dims; i++) {
        if (i == param->axis) {
            begin_mask.push_back(0);
            end_mask.push_back(0);
        } else {
            begin_mask.push_back(1);
            end_mask.push_back(1);
        }
        begins.push_back(0);
        ends.push_back(0);
    }

    ngraph::NodeVector outputNodes;
    
    for (int i = 0; i < param->slices.size(); i++) {
        begins[param->axis] = ends[param->axis];
        ends[param->axis]  += param->slices[i];

        auto beginNode = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::i32, dims_shape, begins);
        auto endNode   = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::i32, dims_shape, ends);
        
        auto strideSliceNode = std::make_shared<ngraph::op::v1::StridedSlice>(
            input_node->output(0), beginNode, endNode, begin_mask, end_mask);
        
        outputNodes.push_back(strideSliceNode);
    }

    SetOutputTensors(outputNodes);

    return TNN_OK;

}

REGISTER_OPENVINO_LAYER_BUILDER(Splitv, LAYER_SPLITV);

}