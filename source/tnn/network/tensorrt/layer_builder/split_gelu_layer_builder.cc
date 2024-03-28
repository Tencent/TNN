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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(SplitGELU, LAYER_FUSED_SPLIT_GELU);

bool SplitGELUTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    if (pos == 0) {
        return (inOut[0].type == nvinfer1::DataType::kFLOAT || inOut[0].type == nvinfer1::DataType::kHALF) &&
               inOut[0].format == nvinfer1::TensorFormat::kLINEAR;
    }
    return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
}

Status SplitGELUTRTPluginLayerBuilder::Reshape() {
    return m_layer->Reshape();
}

const char* SplitGELUTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "SplitGELU";
}

nvinfer1::DataType SplitGELUTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* SplitGELUTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs SplitGELUTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                                           nvinfer1::IExprBuilder& exprBuilder) noexcept {
    DimsExprs output(inputs[0]);
    output.d[2] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[0].d[2], *exprBuilder.constant(2));
    return output;
}

const char* SplitGELUPluginCreator::getPluginName() const noexcept {
    return "SplitGELU";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(SplitGELU, LAYER_FUSED_SPLIT_GELU);

}  //  namespace TNN_NS
