// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Cumsum, LAYER_CUMSUM);

bool CumsumTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF ||
             inOut[pos].type == nvinfer1::DataType::kINT32) && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

Status CumsumTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* CumsumTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "Cumsum";
}

nvinfer1::DataType CumsumTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* CumsumTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    // TODO: Support TRT LayerBuilder instead of Plugin
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs CumsumTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    DimsExprs output(inputs[0]);
    for (int i = 1; i < nbInputs; i++) {
        for (int j = 0; j < output.nbDims; j++) {
            output.d[j] = exprBuilder.operation(DimensionOperation::kMAX, *output.d[j], *inputs[i].d[j]);
        }
    }
    auto layer_param = dynamic_cast<CumsumLayerParam*>(param_);
    if (layer_param->exclusive_extend) {
        output.d[layer_param->axis] = exprBuilder.operation(DimensionOperation::kSUM, *output.d[layer_param->axis], *exprBuilder.constant(1));
    }
    return output;
}

const char* CumsumPluginCreator::getPluginName() const noexcept {
    return "Cumsum";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Cumsum, LAYER_CUMSUM);

}  //  namespace TNN_NS
