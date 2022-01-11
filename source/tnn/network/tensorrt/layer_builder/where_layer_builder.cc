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

#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"
#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(Where, LAYER_WHERE);

ILayer* WhereTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto input_tensors = GetInputITensors();
    auto x = input_tensors[0];
    auto y = input_tensors[1];
    auto condition = input_tensors[2];

    if (condition->getType() == nvinfer1::DataType::kFLOAT) {
        ILayer* cast_layer = network->addIdentity(*condition);
        cast_layer->setOutputType(0, nvinfer1::DataType::kBOOL);
        condition = cast_layer->getOutput(0);
    }

    BroadcastTensors(network, x, y, condition);

    ISelectLayer* layer = network->addSelect(*condition, *x, *y);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }
    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Where, LAYER_WHERE);

}  //  namespace TNN_NS
