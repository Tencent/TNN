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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(PadV2, LAYER_PADV2);

bool PadV2TRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return inOut[pos].type == nvinfer1::DataType::kFLOAT;
}

const char* PadV2TRTPluginLayerBuilder::getPluginType() const {
    return "PadV2";
}

nvinfer1::DataType PadV2TRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* PadV2TRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs PadV2TRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInput, nvinfer1::IExprBuilder& exprBuilder) {
    DimsExprs output(inputs[0]);
    auto param = dynamic_cast<PadLayerParam*>(param_);
    int dim_size = output.nbDims;
    for(int i = 0; i < dim_size; ++i) {
        auto sum = exprBuilder.constant(param->pads[i] + param->pads[dim_size + i]);
        output.d[i] = exprBuilder.operation(DimensionOperation::kSUM, *output.d[i], *sum);
    }
    return output;
}

const char* PadV2PluginCreator::getPluginName() const {
    return "PadV2";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(PadV2, LAYER_PADV2);

}  //  namespace TNN_NS
