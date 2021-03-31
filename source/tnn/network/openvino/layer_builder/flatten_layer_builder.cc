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

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(Flatten, LAYER_FLATTEN);

Status FlattenOVLayerBuilder::Build() {
    if (GetInputNodes().size() <= 0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    auto paramlist = dynamic_cast<FlattenLayerParam *>(param_);
    CHECK_PARAM_NULL(paramlist);

    auto get_shape_count = [&](const ngraph::Shape &shape, int axis) -> size_t {
        size_t res = 1;
        for (int i = axis; i < shape.size(); i++)
            res *= shape[i];
        return res;
    };
    size_t m = input_node->get_output_shape(0)[0];
    size_t n = get_shape_count(input_node->get_output_shape(0), 1);

    std::vector<int> flattenShape;
    flattenShape.push_back(m);
    flattenShape.push_back(n);

    auto patternNode =
        std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i32, ngraph::Shape({2}), flattenShape);
    auto reshapeNode = std::make_shared<ngraph::op::v1::Reshape>(input_node->output(0), patternNode, true);

    reshapeNode->set_friendly_name(param_->name);
    reshapeNode->validate_and_infer_types();

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(reshapeNode);
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(Flatten, LAYER_FLATTEN);

}  // namespace TNN_NS