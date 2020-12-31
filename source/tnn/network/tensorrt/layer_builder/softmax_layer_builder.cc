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

#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Softmax, LAYER_SOFTMAX);

bool SoftmaxTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW
        && inOut[pos].type == inOut[0].type);
}

const char* SoftmaxTRTPluginLayerBuilder::getPluginType() const {
    return "Softmax";
}

nvinfer1::DataType SoftmaxTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* SoftmaxTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<SoftmaxLayerParam*>(param_);
    if (paramlist->axis == 1) {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
        auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
        ISoftMaxLayer* layar = network->addSoftMax(*tensor);
        if (layar != nullptr) {
            layar->setName(layer_name_.c_str());
        }
        return layar;
    } else {
        return TensorRTPluginLayerBuilder::AddToNetwork(network);
    }
}

const char* SoftmaxPluginCreator::getPluginName() const {
    return "Softmax";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Softmax, LAYER_SOFTMAX);

}  //  namespace TNN_NS