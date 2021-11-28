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

DECLARE_OPENVINO_LAYER_BUILDER(PriorBox, LAYER_PRIOR_BOX);

Status PriorBoxOVLayerBuilder::Build() {
    
    auto paramlist = dynamic_cast<PriorBoxLayerParam*>(param_);

    if (GetInputNodes().size() > 2) {
        LOGE("Error: Prior box requires 1 or 2 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes();

    ngraph::op::PriorBoxAttrs attrs;

    attrs.min_size      = paramlist->min_sizes;
    attrs.max_size      = paramlist->max_sizes;
    attrs.aspect_ratio  = paramlist->aspect_ratios;
    attrs.density;                                  // miss
    attrs.fixed_ratio;                              // miss
    attrs.fixed_size;                               // miss
    attrs.clip          = paramlist->clip;
    attrs.flip          = false; // paramlist->flip; flip keep false
    attrs.step          = paramlist->step_h;        // step_w
    attrs.offset        = paramlist->offset;
    attrs.variance      = paramlist->variances;
    attrs.scale_all_sizes = true;                          // miss

    auto size_shape = input_node[0]->get_output_shape(0);
    auto sizeNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i64, ngraph::Shape({2}), std::vector<size_t>({size_shape.at(2), size_shape.at(3)}));
    
    std::vector<size_t> image_shape;
    if (paramlist->img_h == 0 || paramlist->img_w == 0) {
        image_shape.push_back(input_node[1]->get_output_shape(0)[2]);
        image_shape.push_back(input_node[1]->get_output_shape(0)[3]);
    } else {
        image_shape.push_back(paramlist->img_h);
        image_shape.push_back(paramlist->img_w);
    }
    auto imageNode = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i64, ngraph::Shape({2}), image_shape);

    auto priorBoxNode = std::make_shared<ngraph::op::PriorBox>(
        sizeNode->output(0), imageNode->output(0), attrs);
    
    ngraph::Shape reshape;
    reshape.push_back(1);
    for (auto shape : priorBoxNode->get_output_shape(0)) {
        reshape.push_back(shape);
    }
    reshape.push_back(1);
    auto reshapeConst = std::make_shared<ngraph::op::Constant>(
        ngraph::element::Type_t::i32, ngraph::Shape({reshape.size()}), reshape);
    auto reshapeNode = std::make_shared<ngraph::op::v1::Reshape>(
        priorBoxNode->output(0), reshapeConst, false);
    reshapeNode->set_friendly_name(paramlist->name);
    ngraph::NodeVector outputNodes;
    outputNodes.push_back(reshapeNode);
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(PriorBox, LAYER_PRIOR_BOX);

}