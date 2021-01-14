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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

bool StrideSliceV2TRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW
        && inOut[pos].type == inOut[0].type);
}

const char* StrideSliceV2TRTPluginLayerBuilder::getPluginType() const {
    return "StrideSliceV2";
}

nvinfer1::DataType StrideSliceV2TRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* StrideSliceV2TRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs StrideSliceV2TRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    StrideSliceV2LayerParam* param = dynamic_cast<StrideSliceV2LayerParam*>(param_);
    DimsExprs output(inputs[0]);
    for (int i = 0; i < param->axes.size(); i++) {
        int index = param->axes[i];
        output.d[index] = exprBuilder.constant((param->ends[i] - param->begins[i] - 1) / param->strides[i] + 1);
    }
    return output;
}

const char* StrideSliceV2PluginCreator::getPluginName() const {
    return "StrideSliceV2";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

}  //  namespace TNN_NS
