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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(ReduceL2, LAYER_REDUCE_L2);

bool ReduceL2TRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept{
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

Status ReduceL2TRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* ReduceL2TRTPluginLayerBuilder::getPluginType() const noexcept {
    return "ReduceL2";
}

nvinfer1::DataType ReduceL2TRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* ReduceL2TRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs ReduceL2TRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    auto param = dynamic_cast<ReduceLayerParam*>(param_);
    DimsExprs output;
    if (param->keep_dims == 0) {
        int index = 0;
        for (int i = 0; i < inputs[0].nbDims; i++) {
            if (std::find(param->axis.begin(), param->axis.end(), i) == param->axis.end()) {
                output.d[index++] = inputs[0].d[i];
            }
        }
        output.nbDims = index;
    } else {
        for (int i = 0; i < inputs[0].nbDims; i++) {
            output.d[i] = inputs[0].d[i];
        }
        output.nbDims = inputs[0].nbDims;
        for (auto& axis : param->axis) {
            output.d[axis] = exprBuilder.constant(1);
        }
    }

    return output;
}

const char* ReduceL2PluginCreator::getPluginName() const noexcept {
    return "ReduceL2";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(ReduceL2, LAYER_REDUCE_L2);

}  //  namespace TNN_NS
