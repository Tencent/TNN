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

DECLARE_TENSORRT_LAYER_BUILDER(Deconvolution, LAYER_DECONVOLUTION);

ILayer* DeconvolutionTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<ConvLayerParam*>(param_);

    if (paramlist->dialations[1] != 1 || paramlist->dialations[0] != 1) {
        LOGE("TRT does not support dilated deconvolutions");
        return nullptr;
    }
    auto resource = dynamic_cast<ConvLayerResource*>(resource_);
    Weights kernelWeights;
    kernelWeights.type = nvinfer1::DataType::kFLOAT;
    kernelWeights.values = resource->filter_handle.force_to<void*>();
    kernelWeights.count = resource->filter_handle.GetDataCount();

    Weights biasWeights;
    biasWeights.type = nvinfer1::DataType::kFLOAT;
    if (paramlist->bias) {
        biasWeights.values = resource->bias_handle.force_to<void*>();
        biasWeights.count = resource->bias_handle.GetDataCount();
    } else {
        biasWeights.values = nullptr;
        biasWeights.count = 0;
    }

    ILayer* last_layer;
    DimsHW kernalSize(paramlist->kernels[1], paramlist->kernels[0]);
    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
    IDeconvolutionLayer* deconv_layer = network->addDeconvolution(*tensor, paramlist->output_channel,
        kernalSize, kernelWeights, biasWeights);
    if (deconv_layer != nullptr) {
        deconv_layer->setName(layer_name_.c_str());
        deconv_layer->setStride(DimsHW(paramlist->strides[1], paramlist->strides[0]));
        deconv_layer->setPadding(DimsHW(paramlist->pads[2], paramlist->pads[0]));
        deconv_layer->setNbGroups(paramlist->group);
        if (paramlist->pad_type == -1) {
            deconv_layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_DOWN);
        } else {
            deconv_layer->setPaddingMode(PaddingMode::kSAME_LOWER);
        }
        last_layer = deconv_layer;
    }

    IActivationLayer* activation_layer;
    if (paramlist->activation_type == ActivationType_ReLU) {
        activation_layer = network->addActivation(*(deconv_layer->getOutput(0)), nvinfer1::ActivationType::kRELU);
        last_layer = activation_layer;
    } else if (paramlist->activation_type == ActivationType_ReLU6) {
        activation_layer = network->addActivation(*(deconv_layer->getOutput(0)), nvinfer1::ActivationType::kCLIP);
        activation_layer->setAlpha(0.f);
        activation_layer->setBeta(6.f);
        last_layer = activation_layer;
    } else if (paramlist->activation_type != ActivationType_None) {
        LOGE("Error: Unsupport reshape type(%d)", paramlist->activation_type);
        return nullptr;
    }

    return last_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Deconvolution, LAYER_DECONVOLUTION);

}  //  namespace TNN_NS
