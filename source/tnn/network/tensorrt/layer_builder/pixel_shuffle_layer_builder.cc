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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(PixelShuffle, LAYER_PIXEL_SHUFFLE);

bool PixelShuffleTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW
        && inOut[pos].type == inOut[0].type);
}

Status PixelShuffleTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* PixelShuffleTRTPluginLayerBuilder::getPluginType() const {
    return "PixelShuffle";
}

nvinfer1::DataType PixelShuffleTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* PixelShuffleTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs PixelShuffleTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    auto param = dynamic_cast<PixelShuffleLayerParam*>(param_);
    DimsExprs output(inputs[0]);
    auto upscale_factor = exprBuilder.constant(param->upscale_factor);
    auto upscale_factor_square = exprBuilder.constant(param->upscale_factor * param->upscale_factor);
    output.d[1] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[0].d[1], *upscale_factor_square);
    output.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *upscale_factor);
    output.d[3] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[3], *upscale_factor);
    return output;
}

const char* PixelShufflePluginCreator::getPluginName() const {
    return "PixelShuffle";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(PixelShuffle, LAYER_PIXEL_SHUFFLE);

}  //  namespace TNN_NS
