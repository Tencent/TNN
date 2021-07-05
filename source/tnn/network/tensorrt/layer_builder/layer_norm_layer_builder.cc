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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(LayerNorm, LAYER_LAYER_NORM);

bool LayerNormTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    bool layout_check = inOut[pos].format == nvinfer1::TensorFormat::kNCHW;

    bool datatype_check = true;
    if (pos == 0) {
        datatype_check = inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF;
    } else {
        datatype_check = inOut[pos].type == inOut[0].type;
    }

    return layout_check && datatype_check;
}

Status LayerNormTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* LayerNormTRTPluginLayerBuilder::getPluginType() const {
    return "LayerNorm";
}

nvinfer1::DataType LayerNormTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* LayerNormTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs LayerNormTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
}

const char* LayerNormPluginCreator::getPluginName() const {
    return "LayerNorm";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(LayerNorm, LAYER_LAYER_NORM);

}  //  namespace TNN_NS
