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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Squeeze, LAYER_SQUEEZE);

bool SqueezeTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {

    return (inOut[pos].type == nvinfer1::DataType::kFLOAT && 
            inOut[pos].format == nvinfer1::TensorFormat::kNCHW && 
            inOut[pos].type == inOut[0].type ) || 
            inOut[pos].type == nvinfer1::DataType::kINT32;
}

const char* SqueezeTRTPluginLayerBuilder::getPluginType() const {
    return "Squeeze";
}

nvinfer1::DataType SqueezeTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* SqueezeTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs SqueezeTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    auto param = dynamic_cast<SqueezeLayerParam*>(param_);
    auto axes = param->axes;
    DimsExprs output;
    output.nbDims = inputs[0].nbDims - axes.size();
    for (auto& axis : axes) {
        axis = axis < 0 ? axis + inputs[0].nbDims : axis;
    }

    int out_index = 0;
    for (int i = 0; i < inputs[0].nbDims; i++) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            output.d[out_index++] = inputs[0].d[i];
        }
    }
    return output;
}

const char* SqueezePluginCreator::getPluginName() const {
    return "Squeeze";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Squeeze, LAYER_SQUEEZE);


}  //  namespace TNN_NS
