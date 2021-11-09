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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(MatMul, LAYER_MATMUL);

Status MatMulOVLayerBuilder::Build() {
    auto paramlist = dynamic_cast<MatMulLayerParam *>(param_);
    auto resource  = dynamic_cast<MatMulLayerResource *>(resource_);

    if (GetInputNodes().size() <= 0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }

    auto input_node  = GetInputNodes()[0];
    auto input_shape = input_node->get_output_shape(0);

    std::shared_ptr<ngraph::Node> matmul_node;
    auto input_nodes = GetInputNodes();
    if (input_nodes.size() == 2) {
        matmul_node = std::make_shared<ngraph::op::MatMul>(input_nodes[0], input_nodes[1], false, false);
    } else {
        auto weight_dims = paramlist->weight_position == 0 ? paramlist->matrix_a_dims : paramlist->matrix_b_dims;
        auto reshape_const_node =
            std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i32, ngraph::Shape({weight_dims.size()}), weight_dims);

        auto weight_node = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::f32,
            ngraph::Shape({1, static_cast<uint64_t>(DimsVectorUtils::Count(weight_dims))}),
            resource->weight.force_to<float *>());
        auto weight_reshape_node =
            std::make_shared<ngraph::op::v1::Reshape>(weight_node->output(0), reshape_const_node, true);

        if (paramlist->weight_position == 0) {
            matmul_node = std::make_shared<ngraph::op::MatMul>(weight_reshape_node, input_nodes[0], false, false);
        } else {
            matmul_node = std::make_shared<ngraph::op::MatMul>(input_nodes[0], weight_reshape_node, false, false);
        }
    }

    matmul_node->set_friendly_name(paramlist->name);
    matmul_node->validate_and_infer_types();

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(matmul_node);
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(MatMul, LAYER_MATMUL);
}  // namespace TNN_NS