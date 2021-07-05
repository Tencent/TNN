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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(StrideSlice, LAYER_STRIDED_SLICE);

bool StrideSliceTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW
        && inOut[pos].type == inOut[0].type);
}

Status StrideSliceTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* StrideSliceTRTPluginLayerBuilder::getPluginType() const {
    return "StrideSlice";
}

nvinfer1::DataType StrideSliceTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* StrideSliceTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs StrideSliceTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    StrideSliceLayerParam* param = dynamic_cast<StrideSliceLayerParam*>(param_);
    auto begins  = param->begins;
    auto ends    = param->ends;
    auto strides = param->strides;
    std::reverse(begins.begin(), begins.end());
    std::reverse(ends.begin(), ends.end());
    std::reverse(strides.begin(), strides.end());
    DimsExprs output(inputs[0]);
    if (nbInputs == 1) {
        for (int i = 0; i < inputs[0].nbDims; i++) {
            output.d[i] = exprBuilder.constant((ends[i] - begins[i] - 1) / strides[i] + 1);
        }
    }

    return output;
}

const char* StrideSlicePluginCreator::getPluginName() const {
    return "StrideSlice";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(StrideSlice, LAYER_STRIDED_SLICE);

}  //  namespace TNN_NS
