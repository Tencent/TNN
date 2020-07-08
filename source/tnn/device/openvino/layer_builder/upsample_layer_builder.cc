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
#include <ngraph/opsets/opset3.hpp>
#include <inference_engine.hpp>

#include "tnn/layer/base_layer.h"
#include "tnn/device/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/device/openvino/openvino_types.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(Upsample, LAYER_UPSAMPLE);

Status UpsampleOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<UpsampleLayerParam*>(param_);

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    ngraph::op::v0::InterpolateAttrs attrs;
    attrs.align_corners = paramlist->align_corners;
    // if (paramlist->align_corners) 
    //     attrs.coordinate_transformation_mode = ngraph::op::v3::Interpolate::CoordinateTransformMode::align_corners;
    for (size_t axis = 2; axis < input_node->get_output_shape(0).size(); axis++) {
        attrs.axes.insert(axis);
    }

    if (paramlist->type == 1) {
        attrs.mode = "nearest";
    } else {
        attrs.mode = "linear";
    }

    std::vector<int64_t> upsampleShape;
    upsampleShape.push_back(input_node->get_output_shape(0).at(2) * paramlist->scales.at(1));
    upsampleShape.push_back(input_node->get_output_shape(0).at(3) * paramlist->scales.at(0));
    auto upsampleConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i64, ngraph::Shape{2}, upsampleShape);

    auto upsampleNode = std::make_shared<ngraph::op::v0::Interpolate>(
        input_node->output(0), upsampleConst, attrs);
    upsampleNode->validate_and_infer_types();

    upsampleNode->set_friendly_name(paramlist->name);
    
    ngraph::NodeVector outputNodes;
    outputNodes.push_back(upsampleNode);
    SetOutputNodes(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Upsample, LAYER_UPSAMPLE);

}