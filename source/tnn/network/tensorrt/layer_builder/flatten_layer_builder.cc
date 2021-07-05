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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Flatten, LAYER_FLATTEN);

bool FlattenTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kINT32;
}

Status FlattenTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* FlattenTRTPluginLayerBuilder::getPluginType() const {
    return "Flatten";
}

nvinfer1::DataType FlattenTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* FlattenTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs FlattenTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputDims, nvinfer1::IExprBuilder& exprBuilder) {
    DimsExprs output;
    output.nbDims = 2;
    output.d[0] = exprBuilder.constant(1);
    output.d[1] = exprBuilder.constant(1);
    auto *layer_param = dynamic_cast<FlattenLayerParam *>(param_);
    int axis = layer_param->axis;
    for (int i = 0; i < axis; i++) {
        output.d[0] = exprBuilder.operation(DimensionOperation::kPROD, *output.d[0], *inputs[0].d[i]);
    }

    for (int i = axis; i < inputs[0].nbDims; i++) {
        output.d[1] = exprBuilder.operation(DimensionOperation::kPROD, *output.d[1], *inputs[0].d[i]);
    }

    return output;
}

const char* FlattenPluginCreator::getPluginName() const {
    return "Flatten";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Flatten, LAYER_FLATTEN);

}  //  namespace TNN_NS
