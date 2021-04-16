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

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/layer/base_layer.h"
#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/network/openvino/openvino_types.h"
#include "tnn/network/openvino/utils.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(Gather, LAYER_GATHER);

Status GatherOVLayerBuilder::Build() {
    auto paramlist = dynamic_cast<GatherLayerParam *>(param_);
    auto resource  = dynamic_cast<GatherLayerResource *>(resource_);

    if ((paramlist->data_in_resource || paramlist->indices_in_resource) && !resource) {
        LOGE("Gather resource is invalid");
        return TNNERR_INIT_LAYER;
    }

    std::shared_ptr<ngraph::op::Constant> const_node = nullptr;
    if (paramlist->data_in_resource) {
        const_node = ConvertToConstNode(&resource->data);
    } else if (paramlist->indices_in_resource) {
        const_node = ConvertToConstNode(&resource->indices);
    } else {
        return TNNERR_PARAM_ERR;
    }

    if (GetInputNodes().size() <= 0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];
    auto axis_node  = std::make_shared<ngraph::op::Constant>(ngraph::element::i32, ngraph::Shape(), paramlist->axis);

    std::shared_ptr<ngraph::op::v1::Gather> gatherNode = nullptr;
    if (paramlist->data_in_resource) {
        gatherNode = std::make_shared<ngraph::op::v1::Gather>(const_node->output(0), input_node->output(0),
                                                              axis_node->output(0));
    } else {
        gatherNode = std::make_shared<ngraph::op::v1::Gather>(input_node->output(0), const_node->output(0),
                                                              axis_node->output(0));
    }

    gatherNode->validate_and_infer_types();
    gatherNode->set_friendly_name(paramlist->name);

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(gatherNode);
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Gather, LAYER_GATHER);

}  // namespace TNN_NS