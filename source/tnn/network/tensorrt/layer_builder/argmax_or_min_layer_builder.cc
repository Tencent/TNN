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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

bool ArgMaxOrMinTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    if (pos == 0) {
        return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW);
    } else {
        return ((inOut[pos].type == nvinfer1::DataType::kINT32) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW);
    }
}

Status ArgMaxOrMinTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* ArgMaxOrMinTRTPluginLayerBuilder::getPluginType() const {
    return "ArgMaxOrMin";
}

nvinfer1::DataType ArgMaxOrMinTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return nvinfer1::DataType::kINT32;
}

ILayer* ArgMaxOrMinTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs ArgMaxOrMinTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    auto param = dynamic_cast<ArgMaxOrMinLayerParam*>(param_);
    DimsExprs output(inputs[0]);
    output.d[param->axis] = exprBuilder.constant(1);
    if (!param->keep_dims) {
        for (int i = param->axis; i < output.nbDims-1; i++) {
            output.d[i] = output.d[i+1];
        }
        output.nbDims = output.nbDims - 1;
    }
    return output;
}

const char* ArgMaxOrMinPluginCreator::getPluginName() const {
    return "ArgMaxOrMin";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

}  //  namespace TNN_NS
