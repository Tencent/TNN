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

#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Pooling1D, LAYER_POOLING_1D);

bool Pooling1DTRTPluginLayerBuilder::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
}

Status Pooling1DTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* Pooling1DTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "Pooling1D";
}

nvinfer1::DataType Pooling1DTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInpouts) const noexcept {
    return inputTypes[0];
}

ILayer* Pooling1DTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs Pooling1DTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    auto paramlist = dynamic_cast<PoolingLayerParam*>(param_);
    DimsExprs output(inputs[0]);
    output.d[2] = exprBuilder.constant(GetOutputBlobs()[0]->GetBlobDesc().dims[2]);
    return output;
}

const char* Pooling1DPluginCreator::getPluginName() const noexcept {
    return "Pooling1D";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Pooling1D, LAYER_POOLING_1D);

}