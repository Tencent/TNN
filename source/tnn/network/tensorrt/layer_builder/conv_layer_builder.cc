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

DECLARE_TENSORRT_LAYER_BUILDER(Convolution, LAYER_CONVOLUTION);

ILayer* ConvolutionTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<ConvLayerParam*>(param_);
    auto resource = dynamic_cast<ConvLayerResource*>(resource_);

    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
    bool int8 = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetInt8Mode();
    Weights kernelWeights;
    float* tmp = (float*)malloc(resource->filter_handle.GetDataCount() * sizeof(float));
    float scale = *(resource->scale_handle.force_to<float*>());
    for (int i = 0; i < resource->filter_handle.GetDataCount(); i++) {
        tmp[i] = (resource->filter_handle.force_to<float*>())[i] * scale;
    }
    kernelWeights.type = nvinfer1::DataType::kFLOAT;
    kernelWeights.values = tmp;//resource->filter_handle.force_to<void*>();
    kernelWeights.count = resource->filter_handle.GetDataCount();

    Weights biasWeights;
    biasWeights.type = nvinfer1::DataType::kFLOAT;
    if (paramlist->bias) {
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
    
    DimsHW kernelSize(paramlist->kernels[1], paramlist->kernels[0]);
    IConvolutionLayer* layer;
    if (paramlist->pad_type == -1) {
        layer = network->addConvolution(*tensor, paramlist->output_channel, kernelSize, kernelWeights, biasWeights);
        if (layer != nullptr) {
            layer->setName(layer_name_.c_str());
            layer->setStride(DimsHW(paramlist->strides[1], paramlist->strides[0]));
            layer->setDilation(DimsHW(paramlist->dialations[1], paramlist->dialations[0]));
            layer->setPadding(DimsHW(paramlist->pads[2], paramlist->pads[0]));
            layer->setNbGroups(paramlist->group);
        }
    } else {
        IPaddingLayer* padding_layer = network->addPadding(*tensor, DimsHW{0, 0}, DimsHW{1, 1});
        ITensor* tensor = padding_layer->getOutput(0);
        layer = network->addConvolution(*tensor, paramlist->output_channel, kernelSize, kernelWeights, biasWeights);
        if(layer != NULL) {
            layer->setName(layer_name_.c_str());
            layer->setStride(DimsHW(paramlist->strides[1], paramlist->strides[0]));
            layer->setDilation(DimsHW(paramlist->dialations[1], paramlist->dialations[0]));
            layer->setNbGroups(paramlist->group);
        }
    }
    if (int8) {
        layer->setPrecision(nvinfer1::DataType::kINT8);
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Convolution, LAYER_CONVOLUTION);

}  //  namespace TNN_NS
