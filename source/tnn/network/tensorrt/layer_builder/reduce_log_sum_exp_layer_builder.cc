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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(ReduceLogSumExp, LAYER_REDUCE_LOG_SUM_EXP);

bool ReduceLogSumExpTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) &&
        inOut[pos].format == nvinfer1::TensorFormat::kNCHW && inOut[pos].type == inOut[0].type;
}

Status ReduceLogSumExpTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* ReduceLogSumExpTRTPluginLayerBuilder::getPluginType() const {
    return "ReduceLogSumExp";
}

nvinfer1::DataType ReduceLogSumExpTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* ReduceLogSumExpTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs ReduceLogSumExpTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
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

const char* ReduceLogSumExpPluginCreator::getPluginName() const {
    return "ReduceLogSumExp";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(ReduceLogSumExp, LAYER_REDUCE_LOG_SUM_EXP);

}  //  namespace TNN_NS
