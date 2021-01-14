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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(DetectionOutput, LAYER_DETECTION_OUTPUT);

bool DetectionOutputTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW
        && inOut[pos].type == inOut[0].type);
}

const char* DetectionOutputTRTPluginLayerBuilder::getPluginType() const {
    return "DetectionOutput";
}

nvinfer1::DataType DetectionOutputTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* DetectionOutputTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

const char* DetectionOutputPluginCreator::getPluginName() const {
    return "DetectionOutput";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(DetectionOutput, LAYER_DETECTION_OUTPUT);

}  //  namespace TNN_NS