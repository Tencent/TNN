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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(GroupNorm, LAYER_GROUP_NORM);

bool GroupNormTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    const auto &desc = inOut[pos];
    const auto common_cond = nbInputs == 3 && nbOutputs == 1;
    switch (pos)
    {
    case 0:
        return common_cond 
            && (desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF)
            && desc.format == nvinfer1::TensorFormat::kLINEAR && (pos == 0 || inOut[pos].type == inOut[0].type);
    case 1:
    case 2:
        return common_cond && desc.type == nvinfer1::DataType::kFLOAT;
    case 3:
        return common_cond
            && desc.type == inOut[0].type
            && desc.format == nvinfer1::TensorFormat::kLINEAR;
    default:
        return false;
    }
}

Status GroupNormTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* GroupNormTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "GroupNorm";
}

nvinfer1::DataType GroupNormTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* GroupNormTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs GroupNormTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    DimsExprs output(inputs[0]);
    return output;
}

const char* GroupNormPluginCreator::getPluginName() const noexcept {
    return "GroupNorm";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(GroupNorm, LAYER_GROUP_NORM);

}  //  namespace TNN_NS
