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
#include "tnn/utils/bbox_util.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(DetectionOutput, LAYER_DETECTION_OUTPUT);

Status DetectionOutputOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<DetectionOutputLayerParam*>(param_);

    if (GetInputNodes().size() < 3) {
        LOGE("Error: Detection Output Layer requires 3 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes();

    ngraph::op::DetectionOutputAttrs attrs;
    attrs.num_classes = paramlist->num_classes;
    attrs.background_label_id = paramlist->background_label_id;
    attrs.top_k = paramlist->nms_param.top_k;
    attrs.variance_encoded_in_target = paramlist->variance_encoded_in_target;
    attrs.keep_top_k = std::vector<int>({ paramlist->keep_top_k }); // default

    if (paramlist->code_type == PriorBoxParameter_CodeType_CORNER) {
        attrs.code_type = std::string{"caffe.PriorBoxParameter.CORNOR"};
    } else if (paramlist->code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
        attrs.code_type = std::string{"caffe.PriorBoxParameter.CENTER_SIZE"};
    }

    attrs.normalized = true;
    
    attrs.share_location = paramlist->share_location;
    attrs.nms_threshold = paramlist->nms_param.nms_threshold;
    attrs.confidence_threshold = paramlist->confidence_threshold;
    
    ngraph::Shape loc_shape;
    loc_shape.push_back(input_node[0]->get_output_shape(0).at(0));
    loc_shape.push_back(input_node[0]->get_output_shape(0).at(1));
    auto locConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, ngraph::Shape({loc_shape.size()}), loc_shape);
    auto locNode = std::make_shared<ngraph::op::v1::Reshape>(
        input_node[0]->output(0), locConst, false);

    ngraph::Shape conf_shape;
    conf_shape.push_back(input_node[1]->get_output_shape(0).at(0));
    conf_shape.push_back(input_node[1]->get_output_shape(0).at(1));
    auto confConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, ngraph::Shape({conf_shape.size()}), conf_shape);
    auto confNode = std::make_shared<ngraph::op::v1::Reshape>(
        input_node[1]->output(0), confConst, false);
    
    ngraph::Shape prior_shape;
    prior_shape.push_back(input_node[2]->get_output_shape(0).at(0));
    prior_shape.push_back(input_node[2]->get_output_shape(0).at(1));
    prior_shape.push_back(input_node[2]->get_output_shape(0).at(2));
    auto priorConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, ngraph::Shape({prior_shape.size()}), prior_shape);
    auto priorNode = std::make_shared<ngraph::op::v1::Reshape>(
        input_node[2]->output(0), priorConst, false);

    auto detectionOutputNode = std::shared_ptr<ngraph::op::DetectionOutput>();
    if (input_node.size() >= 5) {
        detectionOutputNode = std::make_shared<ngraph::op::DetectionOutput>(
            input_node[0]->output(0), input_node[1]->output(0), input_node[2]->output(0), \
            input_node[3]->output(0), input_node[4]->output(0), attrs);
    } else {
        detectionOutputNode = std::make_shared<ngraph::op::DetectionOutput>(
            locNode->output(0), confNode->output(0), priorNode->output(0), attrs);
    }
    detectionOutputNode->validate_and_infer_types();

    detectionOutputNode->set_friendly_name(paramlist->name);
    ngraph::NodeVector outputNodes;
    outputNodes.push_back(detectionOutputNode);
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(DetectionOutput, LAYER_DETECTION_OUTPUT);

}