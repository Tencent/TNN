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
    return inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kINT32;
}

Status StrideSliceV2TRTPluginLayerBuilder::Reshape() {
    return m_layer->Reshape();
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

        auto begin = exprBuilder.constant(param->begins[i]);
        if (param->begins[i] < 0) {
            begin = exprBuilder.operation(DimensionOperation::kSUM, *begin, *inputs[0].d[index]);
        }

        auto end = exprBuilder.constant(param->ends[i]);
        if (param->ends[i] == INT_MAX) {
            end = inputs[0].d[index];
        }

        if (param->ends[i] < 0) {
            end = exprBuilder.operation(DimensionOperation::kSUM, *end, *inputs[0].d[index]);
        }
        auto stride = exprBuilder.constant(param->strides[i]);
        auto one = exprBuilder.constant(1);
        auto diff = exprBuilder.operation(DimensionOperation::kSUB, *end, *begin);
        diff = exprBuilder.operation(DimensionOperation::kSUB, *diff, *one);
        output.d[index] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *diff, *stride);
        output.d[index] = exprBuilder.operation(DimensionOperation::kSUM, *output.d[index], *one);
    }
    return output;
}

const char* StrideSliceV2PluginCreator::getPluginName() const {
    return "StrideSliceV2";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

}  //  namespace TNN_NS
