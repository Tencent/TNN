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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(RoiAlign, LAYER_ROIALIGN);

bool RoiAlignTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    if (!(nbInputs == 3 && nbOutputs == 1 && pos < nbInputs + nbOutputs)) return false;
    switch (pos) {
        case 0: return inOut[pos].type == nvinfer1::DataType::kFLOAT;
        case 1: return inOut[pos].type == nvinfer1::DataType::kFLOAT;
        case 2: return inOut[pos].type == nvinfer1::DataType::kINT32;
        case 3: return inOut[pos].type == nvinfer1::DataType::kFLOAT;
        default: return false;
    }
}

Status RoiAlignTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* RoiAlignTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "RoiAlign";
}

nvinfer1::DataType RoiAlignTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* RoiAlignTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs RoiAlignTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    RoiAlignLayerParam* param = dynamic_cast<RoiAlignLayerParam*>(param_);

    DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[1].d[0];
    output.d[1] = inputs[0].d[1];
    output.d[2] = exprBuilder.constant(param->output_height);
    output.d[3] = exprBuilder.constant(param->output_width);
    return output;
}

const char* RoiAlignPluginCreator::getPluginName() const noexcept {
    return "RoiAlign";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(RoiAlign, LAYER_ROIALIGN);


}  //  namespace TNN_NS
