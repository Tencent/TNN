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

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(Flatten, LAYER_FLATTEN);

ILayer* FlattenTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto layer_param = dynamic_cast<FlattenLayerParam*>(param_);
    auto tensor = GetInputITensors()[0];

    int axis = layer_param->axis;
    if (axis < 0) axis += tensor->getDimensions().nbDims;

    auto dims = shapeOf(*tensor);
    auto d0 = product(network, dims, 0, axis, 1);
    auto d1 = product(network, dims, axis, dims.size(), 1);

    IShuffleLayer* flatten_layer = addShuffle(network, *tensor, concat(network, d0, d1), false);
    if (flatten_layer != nullptr) {
        flatten_layer->setName(layer_name_.c_str());
    }
    return flatten_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Flatten, LAYER_FLATTEN);

}  //  namespace TNN_NS

