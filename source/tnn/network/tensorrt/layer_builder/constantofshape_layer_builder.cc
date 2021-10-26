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

DECLARE_TENSORRT_LAYER_BUILDER(ConstantOfShape, LAYER_CONSTANT_OF_SHAPE);

ILayer* ConstantOfShapeTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto layer_resource = dynamic_cast<ConstantOfShapeLayerResource*>(resource_);

    auto input_tensors = GetInputITensors();
    int rank = input_tensors[0]->getDimensions().d[0];

    Dims strides{rank};
    std::fill(strides.d, strides.d + strides.nbDims, 0);

    Weights const_weight;
    const_weight = ConvertToWeights(&(layer_resource->value));

    auto weightDims = ConvertToTRTDims(layer_resource->value.GetBufferDims());
    ILayer* constant_layer = network->addConstant(weightDims, const_weight);
    nvinfer1::Dims unsqueezeDims{rank};
    std::fill(unsqueezeDims.d, unsqueezeDims.d + unsqueezeDims.nbDims, 1);
    IShuffleLayer* unsqueeze = network->addShuffle(*constant_layer->getOutput(0));
    unsqueeze->setReshapeDimensions(unsqueezeDims);

    Dims starts;
    starts.nbDims = rank;
    for (int i = 0; i < rank; i++) {
        starts.d[i] = 0;
    }
    ISliceLayer* broadcast_layer = network->addSlice(*unsqueeze->getOutput(0), starts,
        nvinfer1::Dims{}, strides);

    if (broadcast_layer != nullptr) {
        broadcast_layer->setName(layer_name_.c_str());   
        broadcast_layer->setInput(2, *input_tensors[0]);
    }

    return broadcast_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(ConstantOfShape, LAYER_CONSTANT_OF_SHAPE);

}  //  namespace TNN_NS
