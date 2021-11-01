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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Reorg, LAYER_REORG);

bool ReorgTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

Status ReorgTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* ReorgTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "Reorg";
}

nvinfer1::DataType ReorgTRTPluginLayerBuilder::getOutputDataType(int index,
        const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* ReorgTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs ReorgTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    auto param = dynamic_cast<ReorgLayerParam*>(param_);
    DimsExprs output(inputs[0]);
    auto upscale_factor = exprBuilder.constant(param->stride);
    auto upscale_factor_square = exprBuilder.constant(param->stride * param->stride);
    output.d[1] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[0].d[1], *upscale_factor_square);
    output.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *upscale_factor);
    output.d[3] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[3], *upscale_factor);
    return output;
}

const char* ReorgPluginCreator::getPluginName() const noexcept {
    return "Reorg";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Reorg, LAYER_REORG);

}  //  namespace TNN_NS
