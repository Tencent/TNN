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
#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/network/openvino/openvino_types.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(Upsample, LAYER_UPSAMPLE);

Status UpsampleOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<UpsampleLayerParam*>(param_);

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    ngraph::op::v4::Interpolate::InterpolateAttrs attrs;
    if (paramlist->align_corners) {
        attrs.coordinate_transformation_mode = ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners;
    } else {
        attrs.coordinate_transformation_mode = ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel;
    }
    attrs.nearest_mode = ngraph::op::v4::Interpolate::NearestMode::floor;
    // attrs.align_corners = paramlist->align_corners;
    // if (paramlist->align_corners) 
    //     attrs.coordinate_transformation_mode = ngraph::op::v3::Interpolate::CoordinateTransformMode::align_corners;
    std::vector<int64_t> axes;
    for (size_t axis = 2; axis < input_node->get_output_shape(0).size(); axis++) {
        axes.push_back(axis);
    }
    auto axesConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i64, ngraph::Shape{input_node->get_output_shape(0).size() - 2}, axes);

    if (paramlist->mode == 1) {
        attrs.mode = ngraph::op::v4::Interpolate::InterpolateMode::nearest; //"nearest";
    } else if (paramlist->mode == 2) {
        attrs.mode = ngraph::op::v4::Interpolate::InterpolateMode::linear;  //"linear";
    } else if (paramlist->mode == 3){
        attrs.mode = ngraph::op::v4::Interpolate::InterpolateMode::cubic;   //"cubic";
    } else {
        return Status(TNNERR_MODEL_ERR, "Error: Upsample dont support resize type");
    }

    std::vector<int64_t> upsampleShape;
    std::vector<float> upsampleScaleShape;
    upsampleShape.push_back(input_node->get_output_shape(0)[2]);
    upsampleShape.push_back(input_node->get_output_shape(0)[3]);
    upsampleScaleShape.push_back(1.0);
    upsampleScaleShape.push_back(1.0);
    if (paramlist->dims.size() != 0) {
        attrs.shape_calculation_mode = ngraph::op::v4::Interpolate::ShapeCalcMode::sizes;
        if (paramlist->dims[0] != 0 && paramlist->dims[1] != 0) {
            upsampleShape[0] = paramlist->dims[1];
            upsampleShape[1] = paramlist->dims[0];
        } else {
            return Status(TNNERR_MODEL_ERR, "Error: Upsample size error");
        }
    } else {
        attrs.shape_calculation_mode = ngraph::op::v4::Interpolate::ShapeCalcMode::scales;
        upsampleScaleShape[0] = paramlist->scales.at(1);
        upsampleScaleShape[1] = paramlist->scales.at(0);
    }
    auto upsampleConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i64, ngraph::Shape{input_node->get_output_shape(0).size() - 2}, upsampleShape);
    auto upsampleScaleConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::f32, ngraph::Shape{input_node->get_output_shape(0).size() - 2}, upsampleScaleShape);

    auto upsampleNode = std::make_shared<ngraph::op::v4::Interpolate>(
        input_node->output(0), upsampleConst, upsampleScaleConst, axesConst, attrs);
    upsampleNode->validate_and_infer_types();

    upsampleNode->set_friendly_name(paramlist->name);
    
    ngraph::NodeVector outputNodes;
    outputNodes.push_back(upsampleNode);
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Upsample, LAYER_UPSAMPLE);

}