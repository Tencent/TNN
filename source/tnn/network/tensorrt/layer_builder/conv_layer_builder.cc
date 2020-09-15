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

    DimsHW kernalSize(paramlist->kernels[1], paramlist->kernels[0]);
    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
    IConvolutionLayer* layer = network->addConvolution(*tensor, paramlist->output_channel, kernalSize, kernelWeights, biasWeights);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
        layer->setStride(DimsHW(paramlist->strides[1], paramlist->strides[0]));
        layer->setDilation(DimsHW(paramlist->dialations[1], paramlist->dialations[0]));
        layer->setPadding(DimsHW(paramlist->pads[2], paramlist->pads[0]));
        layer->setNbGroups(paramlist->group);
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Convolution, LAYER_CONVOLUTION);

}  //  namespace TNN_NS
