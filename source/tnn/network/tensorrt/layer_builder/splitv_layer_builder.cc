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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER_WITH_FUNC(SplitV, LAYER_SPLITV,
                                                void CheckInputShapeTensor(INetworkDefinition* network););

bool SplitVTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    if (pos == 0) {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos >= nbInputs) {
        return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    } else {
        return true;
    }
}

Status SplitVTRTPluginLayerBuilder::Reshape() {
    return m_layer->Reshape();
}

const char* SplitVTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "SplitV";
}

nvinfer1::DataType SplitVTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* SplitVTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs SplitVTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                                           nvinfer1::IExprBuilder& exprBuilder) noexcept {
    auto param = dynamic_cast<SplitVLayerParam*>(param_);
    DimsExprs output(inputs[0]);
    auto axis = param->axis >= 0 ? param->axis : (inputs[0].nbDims + param->axis);
    if (param->is_split_specified) {
        auto slices = std::set<int>(param->slices.begin(), param->slices.end());
        if (slices.size() == 0 || (slices.size() == 1 && slices.count(0))) {
            output.d[axis] = inputs[1].d[index];
        } else {
            output.d[axis] = exprBuilder.constant(param->slices[index]);
        }
    } else {
        output.d[axis] = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *inputs[0].d[axis],
                                                      *exprBuilder.constant(param->slices.size()));
    }
    return output;
}

void SplitVTRTPluginLayerBuilder::CheckInputShapeTensor(INetworkDefinition* network) {
    auto param = dynamic_cast<SplitVLayerParam*>(param_);
    if (param->is_split_specified) {
        auto slices = std::set<int>(param->slices.begin(), param->slices.end());
        if (slices.size() == 0 || (slices.size() == 1 && slices.count(0))) {
            ReplaceInputShapeTensor(1, network);
        }
    }
}

const char* SplitVPluginCreator::getPluginName() const noexcept {
    return "SplitV";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(SplitV, LAYER_SPLITV);
REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(SplitV, LAYER_SPLITTORCH);

}  //  namespace TNN_NS
