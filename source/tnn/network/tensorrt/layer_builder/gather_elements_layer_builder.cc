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

DECLARE_TENSORRT_LAYER_BUILDER(GatherElements, LAYER_GATHERELEMENTS);

ILayer* GatherElementsTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<GatherElementsLayerParam*>(param_);
    auto axis = paramlist->axis;

    if (GetInputITensors().size() != 2) {
        LOGE("GatherElementsTRTLayerBuilder Error, input size not supported");
        return nullptr;
    }

    nvinfer1::ITensor* data = GetInputITensors()[0];
    nvinfer1::ITensor* indices = GetInputITensors()[1];

    int data_rank = data->getDimensions().nbDims;
    int indices_rank = indices->getDimensions().nbDims;

    if (data_rank != indices_rank) {
        LOGE("GatherElementsTRTLayerBuilder Error, data and indices rank not equal");
        return nullptr;
    }

    if (axis < 0) {
        axis += data_rank;
    }

    if (axis >= data_rank) {
        LOGE("GatherElementsTRTLayerBuilder Error, invalid axis");
        return nullptr;
    }

    nvinfer1::IGatherLayer* layer = network->addGather(*data, *indices, axis);
    layer->setName(layer_name_.c_str());

#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR > 80
    layer->setMode(nvinfer1::GatherMode::kELEMENT);
#endif

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(GatherElements, LAYER_GATHERELEMENTS);

}  //  namespace TNN_NS

