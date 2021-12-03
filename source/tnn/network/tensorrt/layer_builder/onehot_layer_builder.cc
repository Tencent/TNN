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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(OneHot, LAYER_ONEHOT);

bool OneHotTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    return ((pos == 0 && inOut[pos].type == nvinfer1::DataType::kINT32) || (pos == 1 && inOut[pos].type == nvinfer1::DataType::kFLOAT));
}

Status OneHotTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* OneHotTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "OneHot";
}

nvinfer1::DataType OneHotTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

ILayer* OneHotTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs OneHotTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    auto param = dynamic_cast<OneHotLayerParam*>(param_);
    DimsExprs output(inputs[0]);
    output.nbDims = output.nbDims + 1;
    int axis = param->axis;
    if(axis < 0) {
        axis += output.nbDims;
    } 
    for (int i = axis; i < output.nbDims - 1; i++) {
            output.d[i + 1] = output.d[i];
    }

    output.d[axis] = exprBuilder.constant(param->depth);
    return output;
}

const char* OneHotPluginCreator::getPluginName() const noexcept {
    return "OneHot";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(OneHot, LAYER_ONEHOT);

}  //  namespace TNN_NS
