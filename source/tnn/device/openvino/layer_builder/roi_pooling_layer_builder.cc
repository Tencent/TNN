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
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <inference_engine.hpp>

#include "tnn/layer/base_layer.h"
#include "tnn/device/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/device/openvino/openvino_types.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(RoiPooling, LAYER_ROIPOOLING);

Status RoiPoolingOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<RoiPoolingLayerParam*>(param_);

    if (GetInputNodes().size() <=1) {
        LOGE("Error: ROI Pooling requires 2 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes();

    ngraph::Shape outputSize;
    std::reverse(paramlist->pooled_dims.begin(), paramlist->pooled_dims.end());
    for (auto item : paramlist->pooled_dims) {
        outputSize.push_back(item);
    }

    std::string pool_type;
    if (paramlist->pool_type == 1) {
        pool_type = "Bilinear";
    } else {
        pool_type = "Max";
    }

    auto roiPoolingNode = std::make_shared<ngraph::op::ROIPooling>(
        input_node[0]->output(0), input_node[1]->output(1), outputSize, paramlist->spatial_scale, pool_type);
    roiPoolingNode->validate_and_infer_types();

    roiPoolingNode->set_friendly_name(paramlist->name);

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(roiPoolingNode);
    SetOutputNodes(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(RoiPooling, LAYER_ROIPOOLING);

}