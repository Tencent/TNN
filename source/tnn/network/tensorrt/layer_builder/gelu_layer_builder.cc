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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Gelu, LAYER_GELU);

bool GeluTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW
        && inOut[pos].type == inOut[0].type);
}

Status GeluTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* GeluTRTPluginLayerBuilder::getPluginType() const {
    return "Gelu";
}

nvinfer1::DataType GeluTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* GeluTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs GeluTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    DimsExprs output(inputs[0]);
    for (int i = 1; i < nbInputs; i++) {
        for (int j = 0; j < output.nbDims; j++) {
            output.d[j] = exprBuilder.operation(DimensionOperation::kMAX, *output.d[j], *inputs[i].d[j]);
        }
    }
    return output;
}

const char* GeluPluginCreator::getPluginName() const {
    return "Gelu";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Gelu, LAYER_GELU);

}  //  namespace TNN_NS
