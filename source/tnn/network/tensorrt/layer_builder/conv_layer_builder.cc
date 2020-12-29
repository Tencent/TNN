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

    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();
    bool int8 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetInt8Mode();

    Weights kernelWeights;
    Weights biasWeights;
    ILayer* last_layer;
    if (int8) {
        float weight_scale_value = *(resource->scale_handle.force_to<float*>());
        float input_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(
            input_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        std::vector<int> dims;
        dims.push_back(paramlist->output_channel);
        dims.push_back(input_blobs_[0]->GetBlobDesc().dims[1] / paramlist->group);
        dims.push_back(paramlist->kernels[1]);
        dims.push_back(paramlist->kernels[0]);
        last_layer = AddInt8WeightQDQLayers(network, &(resource->filter_handle), kernelWeights,
            paramlist->bias ? &(resource->bias_handle) : nullptr, biasWeights,
            1 / (weight_scale_value / input_scale_value), dims);
    } else {
        kernelWeights.type = nvinfer1::DataType::kFLOAT;
        kernelWeights.values = resource->filter_handle.force_to<void*>();
        kernelWeights.count = resource->filter_handle.GetDataCount();
        if (paramlist->bias) {
            biasWeights.type = nvinfer1::DataType::kFLOAT;
            biasWeights.values = resource->bias_handle.force_to<void*>();
            biasWeights.count = resource->bias_handle.GetDataCount();
        } else {
            biasWeights.type = nvinfer1::DataType::kFLOAT;
            biasWeights.values = nullptr;
            biasWeights.count = 0;
        }
    }

    DimsHW kernelSize(paramlist->kernels[1], paramlist->kernels[0]);
    IConvolutionLayer* conv_layer;
    if (paramlist->pad_type == -1) {
        conv_layer = network->addConvolution(*input_tensor, paramlist->output_channel, kernelSize,
            kernelWeights, biasWeights);
        if (int8) conv_layer->setInput(1, *(last_layer->getOutput(0)));
        if (conv_layer != nullptr) {
            conv_layer->setName(layer_name_.c_str());
            conv_layer->setStride(DimsHW(paramlist->strides[1], paramlist->strides[0]));
            conv_layer->setDilation(DimsHW(paramlist->dialations[1], paramlist->dialations[0]));
            conv_layer->setPadding(DimsHW(paramlist->pads[2], paramlist->pads[0]));
            conv_layer->setNbGroups(paramlist->group);
        }
    } else {
        IPaddingLayer* padding_layer = network->addPadding(*input_tensor, DimsHW{0, 0}, DimsHW{1, 1});
        ITensor* pad_tensor = padding_layer->getOutput(0);
        conv_layer = network->addConvolution(*pad_tensor, paramlist->output_channel, kernelSize,
            kernelWeights, biasWeights);
        if (int8) conv_layer->setInput(1, *(last_layer->getOutput(0)));
        if(conv_layer != NULL) {
            conv_layer->setName(layer_name_.c_str());
            conv_layer->setStride(DimsHW(paramlist->strides[1], paramlist->strides[0]));
            conv_layer->setDilation(DimsHW(paramlist->dialations[1], paramlist->dialations[0]));
            conv_layer->setNbGroups(paramlist->group);
        }
    }

    last_layer = conv_layer;

    if (int8) {
        conv_layer->setPrecision(nvinfer1::DataType::kINT8);
    }

    IActivationLayer* activation_layer;
    if (paramlist->activation_type == ActivationType_ReLU) {
        activation_layer = network->addActivation(*(conv_layer->getOutput(0)), nvinfer1::ActivationType::kRELU);
        last_layer = activation_layer;
    } else if (paramlist->activation_type == ActivationType_ReLU6) {
        activation_layer = network->addActivation(*(conv_layer->getOutput(0)), nvinfer1::ActivationType::kCLIP);
        activation_layer->setAlpha(0.f);
        activation_layer->setBeta(6.f);
        last_layer = activation_layer;
    } else if (paramlist->activation_type != ActivationType_None) {
        LOGE("Error: Unsupport reshape type(%d)", paramlist->activation_type);
        return nullptr;
    }

    if (int8) {
        float output_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(
            output_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        return AddInt8OutputQDQLayers(network, last_layer->getOutput(0), output_foreign_tensor,
            output_scale_value, 1 / output_scale_value);
    }

    return last_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Convolution, LAYER_CONVOLUTION);

}  //  namespace TNN_NS

