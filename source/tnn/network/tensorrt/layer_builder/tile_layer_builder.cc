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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Tile, LAYER_REPEAT);

bool TileTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kINT32;
}

Status TileTRTPluginLayerBuilder::Reshape() {
    return m_layer->Reshape();
}

const char* TileTRTPluginLayerBuilder::getPluginType() const {
    return "Tile";
}

nvinfer1::DataType TileTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* TileTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs TileTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputDims, nvinfer1::IExprBuilder& exprBuilder) {
    DimsExprs output;
    auto *layer_param = dynamic_cast<TileLayerParam *>(param_);
    auto reps = layer_param->reps;
    auto input_dims = inputs[0].nbDims;
    int reps_size = reps.size();
    output.nbDims = std::max(reps_size, inputs[0].nbDims);
    int index_i = inputs[0].nbDims-1, index_o = output.nbDims-1, index_r = reps_size-1;
    for (; index_i>=0 && index_r>=0; index_i--, index_o--, index_r--) {
        auto rep = exprBuilder.constant(reps[index_r]);
        output.d[index_o] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[index_i], *rep);
    }

    for (; index_r>=0; index_r--) {
        output.d[index_o] = exprBuilder.constant(reps[index_r]);
    }

    return output;
}

const char* TilePluginCreator::getPluginName() const {
    return "Tile";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Tile, LAYER_REPEAT);

}  //  namespace TNN_NS
