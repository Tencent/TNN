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

DECLARE_OPENVINO_LAYER_BUILDER(ScatterND, LAYER_SCATTER_ND);

Status ScatterNDOVLayerBuilder::Build() {
    auto paramlist = param_;
    auto resource  = dynamic_cast<ScatterNDLayerResource *>(resource_);

    if (!resource && GetInputNodes().size() < 3) {
        LOGE("ScatterNDOVLayerBuilder has not layer resource\n");
        return Status(TNNERR_PARAM_ERR, "ScatterNDOVLayerBuilder has not layer resource");
    }

    auto input_nodes = GetInputNodes();

    std::shared_ptr<ngraph::op::v3::ScatterNDUpdate> scatterNode = nullptr;
    if (input_nodes.size() == 3) {
        scatterNode = std::make_shared<ngraph::op::v3::ScatterNDUpdate>(
            input_nodes[0]->output(0), input_nodes[1]->output(0), input_nodes[2]->output(0));
    } else {
        auto indiceNode = ConvertToConstNode(&resource->indices);
        scatterNode     = std::make_shared<ngraph::op::v3::ScatterNDUpdate>(
            input_nodes[0]->output(0), indiceNode->output(0), input_nodes[1]->output(0));
    }

    scatterNode->validate_and_infer_types();
    scatterNode->set_friendly_name(paramlist->name);

    ngraph::NodeVector outputNodes;
    outputNodes.push_back(scatterNode);
    SetOutputTensors(outputNodes);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(ScatterND, LAYER_SCATTER_ND);

}  // namespace TNN_NS