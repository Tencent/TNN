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
#include "tnn/network/openvino/custom_layer/custom_implementation.h"
#include "tnn/network/openvino/utils.h"

namespace TNN_NS {
DECLARE_CUSTOM_OP(DetectionOutput);
REGISTER_CUSTOM_OP(DetectionOutput);
DECLARE_CUSTOM_IMPLEMENTATION(DetectionOutput);
REGISTER_CUSTOM_IMPLEMENTATION(DetectionOutput, CustomDetectionOutput);
DECLARE_OPENVINO_LAYER_BUILDER(DetectionOutput, LAYER_DETECTION_OUTPUT);

Status DetectionOutputOVLayerBuilder::Build() {
    auto paramlist = dynamic_cast<DetectionOutputLayerParam*>(param_);

    if (GetInputNodes().size() < 3) {
        LOGE("Error: Detection Output Layer requires 3 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes();
    ADD_CUSTOM_NODE(DetectionOutput, paramlist->name);
    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(DetectionOutput, LAYER_DETECTION_OUTPUT);

}