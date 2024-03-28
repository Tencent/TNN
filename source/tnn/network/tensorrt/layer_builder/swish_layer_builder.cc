// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

DECLARE_TENSORRT_LAYER_BUILDER(Swish, LAYER_SWISH);

ILayer* SwishTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    // Swish: y = x * sigmoid(x)
    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();

    // Step 1: Add Activatation Layer (sigmoid)
    IActivationLayer* sigmoid_layer = network->addActivation(*input_tensor, nvinfer1::ActivationType::kSIGMOID);
    if (!sigmoid_layer) {
        LOGE("SwishTRTLayerBuilder: Unable to Add Sigmoid Activation Layer to network.");
        return nullptr;
    }
    sigmoid_layer->setName((layer_name_+"_sigmoid").c_str());
    ITensor* sigmoid_out_tensor = sigmoid_layer->getOutput(0);

    // Step 2: Add Mul Layer
    IElementWiseLayer* mul_layer = network->addElementWise(*input_tensor, *sigmoid_out_tensor, nvinfer1::ElementWiseOperation::kPROD);
    if (!mul_layer) {
        LOGE("SwishTRTLayerBuilder: Unable to Add Mul Layer to network.");
        return nullptr;
    }
    mul_layer->setName((layer_name_+"_mul").c_str());

    return mul_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Swish, LAYER_SWISH);

}  //  namespace TNN_NS
