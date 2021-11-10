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

DECLARE_TENSORRT_LAYER_BUILDER(Convolution1D, LAYER_CONVOLUTION_1D);

ILayer* Convolution1DTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<ConvLayerParam*>(param_);
    auto resource = dynamic_cast<ConvLayerResource*>(resource_);

    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();

    Weights kernelWeights;
    Weights biasWeights;
    ILayer* last_layer;
    kernelWeights = ConvertToWeights(&(resource->filter_handle));
    if (paramlist->bias) {
        biasWeights = ConvertToWeights(&(resource->bias_handle));
    } else {
        biasWeights = ConvertToWeights(nullptr, true, resource->filter_handle.GetDataType());
    }

    Dims kernelSize;
    kernelSize.nbDims = 2;
    kernelSize.d[0] = paramlist->kernels[0];
    kernelSize.d[1] = 1;

    DimsVector unsqueeze_dims(input_tensor->getDimensions().nbDims, 0);
    unsqueeze_dims.push_back(1);
    ILayer* layer = AddReshapeToNetwork(network, input_tensor, unsqueeze_dims, (layer_name_ + "unsqueeze").c_str());

    IConvolutionLayer* conv_layer;
    conv_layer = network->addConvolutionNd(*(layer->getOutput(0)), paramlist->output_channel, kernelSize,
        kernelWeights, biasWeights);
    if (conv_layer != nullptr) {
        conv_layer->setName(layer_name_.c_str());
        Dims strides;
        strides.nbDims = 2;
        strides.d[0] = paramlist->strides[0];
        strides.d[1] = 1;
        conv_layer->setStrideNd(strides);
        Dims dialations;
        dialations.nbDims = 2;
        dialations.d[0] = paramlist->dialations[0];
        dialations.d[1] = 1;
        conv_layer->setDilationNd(dialations);
        Dims pads;
        pads.nbDims = 2;
        pads.d[0] = paramlist->pads[0];
        pads.d[1] = 0;
        conv_layer->setPaddingNd(pads);
        conv_layer->setNbGroups(paramlist->group);
    }

    last_layer = conv_layer;

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

    unsqueeze_dims.erase(unsqueeze_dims.end()-1);
    last_layer = AddReshapeToNetwork(network, last_layer->getOutput(0), unsqueeze_dims, (layer_name_ + "squeeze").c_str());

    return last_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Convolution1D, LAYER_CONVOLUTION_1D);

}  //  namespace TNN_NS

