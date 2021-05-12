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

DECLARE_TENSORRT_LAYER_BUILDER(Gather, LAYER_GATHER);

ILayer* GatherTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    
    auto layer_param = dynamic_cast<GatherLayerParam*>(param_);
    if (layer_param == nullptr) {
        LOGE("GatherTRTLayerBuilder layer_param is null");
        return nullptr;
    }
    int axis = layer_param->axis;
    
    auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);
    if ((layer_param->data_in_resource || layer_param->indices_in_resource) && !layer_resource) {
        LOGE("Gather resource is invalid");
        return nullptr;
    }
    
    nvinfer1::ITensor * data = nullptr;
    nvinfer1::ITensor * indices = nullptr;
    if (layer_param->data_in_resource) {
        auto const_layer = ConvertWeightToConstLayer(network, &(layer_resource->data));
        if (const_layer != nullptr) {
            data = const_layer->getOutput(0);
        }
    } else {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(*input_blobs_.begin())->GetForeignTensor();
        data = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
    }
    
    if (layer_param->indices_in_resource) {
        auto const_layer = ConvertWeightToConstLayer(network, &(layer_resource->indices));
        if (const_layer != nullptr) {
            indices = const_layer->getOutput(0);
        }
    } else {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(*input_blobs_.rbegin())->GetForeignTensor();
        indices = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
    }
    
    if (data == nullptr || indices == nullptr) {
        LOGE("GatherTRTLayerBuilder can not find data or indices\n");
        return nullptr;
    }
    
    return network->addGather(*data, *indices, axis);
}

REGISTER_TENSORRT_LAYER_BUILDER(Gather, LAYER_GATHER);

}  //  namespace TNN_NS