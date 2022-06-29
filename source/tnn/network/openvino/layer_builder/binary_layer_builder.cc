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

#include "tnn/core/macro.h"
#include "tnn/layer/base_layer.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/network/openvino/layer_builder/binary_layer_builder.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/network/openvino/openvino_types.h"
#include "tnn/network/openvino/utils.h"

namespace TNN_NS {
namespace openvino {

Status BinaryLayerBuilder::Build() {

    if (GetInputNodes().size() <= 0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes();

    std::shared_ptr<ngraph::Node> binary_node;
    if (input_node.size() == 2) {
        binary_node = CreateNode(input_node[0]->output(0), input_node[1]->output(0));
    } else {
        auto resource = dynamic_cast<EltwiseLayerResource*>(resource_);
        auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
        CHECK_PARAM_NULL(layer_param);
        const int weight_input_index = layer_param->weight_input_index;

        ngraph::Shape weight_node_shape;
        DimsVector weight_shape = resource->element_shape;
        for(auto d : weight_shape) weight_node_shape.push_back(d);

        auto ov_dtype    = ConvertToOVDataType(resource->element_handle.GetDataType());
        auto weight_node = std::make_shared<ngraph::op::Constant>(ov_dtype, weight_node_shape,
                                                                  resource->element_handle.force_to<float *>());
        weight_node->validate_and_infer_types();

        if (weight_input_index == 0) {
            binary_node = CreateNode(weight_node, input_node[0]->output(0));
        } else {
            binary_node = CreateNode(input_node[0]->output(0), weight_node);
        }
    }
    binary_node->set_friendly_name(param_->name);
    binary_node->validate_and_infer_types();

    SetOutputTensors(ngraph::NodeVector({binary_node}));

    return TNN_OK;
}

} // namespace openvino
} // namespace TNN_NS