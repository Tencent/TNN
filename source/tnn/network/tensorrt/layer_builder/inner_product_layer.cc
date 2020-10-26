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

DECLARE_TENSORRT_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);

ILayer* InnerProductTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<InnerProductLayerParam*>(param_);
    auto resource = dynamic_cast<InnerProductLayerResource*>(resource_);
    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
    bool int8 = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetInt8Mode();

    Weights kernelWeights;
    float* tmp = (float*)malloc(resource->weight_handle.GetDataCount() * sizeof(float));
    float scale = *(resource->scale_handle.force_to<float*>());
    for (int i = 0; i < resource->weight_handle.GetDataCount(); i++) {
        tmp[i] = (resource->weight_handle.force_to<float*>())[i] * scale;
    }
    kernelWeights.type = nvinfer1::DataType::kFLOAT;
    kernelWeights.values = tmp;//resource->weight_handle.force_to<void*>();
    kernelWeights.count = resource->weight_handle.GetDataCount();

    Weights biasWeights;
    biasWeights.type = nvinfer1::DataType::kFLOAT;
    if (paramlist->has_bias) {
        float* tmp2 = (float*)malloc(resource->bias_handle.GetDataCount() * sizeof(float));
        for (int i = 0; i < resource->bias_handle.GetDataCount(); i++) {
            tmp2[i] = (resource->bias_handle.force_to<int*>())[i];
        }
        biasWeights.values = tmp2;//resource->bias_handle.force_to<void*>();
        biasWeights.count = resource->bias_handle.GetDataCount();
    } else {
        biasWeights.values = nullptr;
        biasWeights.count = 0;
    }

    IFullyConnectedLayer* layer = network->addFullyConnected(*tensor, paramlist->num_output, kernelWeights, biasWeights);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }
    if (int8) {
        layer->setPrecision(nvinfer1::DataType::kINT8);
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);

}  //  namespace TNN_NS