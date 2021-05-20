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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Histogram, LAYER_HISTOGRAM);

bool HistogramTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return (inOut[pos].type == nvinfer1::DataType::kINT32);
}

const char* HistogramTRTPluginLayerBuilder::getPluginType() const {
    return "Histogram";
}

nvinfer1::DataType HistogramTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* HistogramTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs HistogramTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    auto layer_param = dynamic_cast<HistogramLayerParam*>(param_);
    DimsExprs output;
    output.nbDims = 1;
    output.d[0] = exprBuilder.constant(layer_param->depth);
    return output;
}

const char* HistogramPluginCreator::getPluginName() const {
    return "Histogram";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Histogram, LAYER_HISTOGRAM);

}  //  namespace TNN_NS
