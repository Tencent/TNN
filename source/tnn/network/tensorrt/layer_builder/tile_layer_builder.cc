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

DECLARE_TENSORRT_LAYER_BUILDER(Tile, LAYER_REPEAT);

ILayer* TileTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<TileLayerParam*>(param_);
    auto reps = paramlist->reps;

    const auto inputDims = shapeOf(*GetInputITensors()[0]);
    ShapeTensor repeats;

    if (GetInputITensors().size() == 2) {
        repeats = ShapeTensor(*GetInputITensors()[1]);
    } else {
        repeats = ShapeTensor(1, std::move(reps));
    }

    ShapeTensor outputShape = mul(network, inputDims, repeats);
    nvinfer1::ISliceLayer* layer = addSlice(network, *GetInputITensors()[0],
        similar(network, inputDims, 0), outputShape, similar(network, inputDims, 1));
    layer->setName(layer_name_.c_str());
    layer->setMode(nvinfer1::SliceMode::kWRAP);

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Tile, LAYER_REPEAT);

}  //  namespace TNN_NS

