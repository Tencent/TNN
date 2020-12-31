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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Pad, LAYER_PAD);

bool PadTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    int channels = inOut[0].dims.d[1];
    bool is_pad_8 = (channels % 8 == 0);
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kNCHW) ||
        (inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == nvinfer1::TensorFormat::kNHWC8 && is_pad_8));
}

const char* PadTRTPluginLayerBuilder::getPluginType() const {
    return "Pad";
}

nvinfer1::DataType PadTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* PadTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

const char* PadPluginCreator::getPluginName() const {
    return "Pad";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Pad, LAYER_PAD);

}  //  namespace TNN_NS