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
#include "tnn/utils/data_type_utils.h"
#include "tnn/network/openvino/custom_layer/custom_batch_norm.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(BatchNorm, LAYER_BATCH_NORM);

Status BatchNormOVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<BatchNormLayerParam*>(param_);

    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }
    auto input_node = GetInputNodes()[0];

    // use custom x86 batch norm layer
    auto batchNormNode = std::make_shared<CustomBatchNormOp>(input_node->outputs(), base_layer_, GetInputBlobs(), GetOutputBlobs());
    batchNormNode->set_friendly_name(param_->name);
    ngraph::NodeVector outputNodes;
    outputNodes.push_back(batchNormNode);
    SetOutputTensors(outputNodes);   

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(BatchNorm, LAYER_BATCH_NORM);

}