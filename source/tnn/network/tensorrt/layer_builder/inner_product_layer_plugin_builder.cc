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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);

bool InnerProductTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kNCHW;
}

Status InnerProductTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* InnerProductTRTPluginLayerBuilder::getPluginType() const {
    return "InnerProduct";
}

nvinfer1::DataType InnerProductTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

DimsExprs InnerProductTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {

    InnerProductLayerParam* ip_param = dynamic_cast<InnerProductLayerParam*>(param_);
    if (!ip_param) {
        LOGE("InnerProductTRTPluginLayerBuilder got null param\n");
        return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
    }

    DimsExprs output(inputs[0]);

    int N    = ip_param->num_output;
    int axis = ip_param->axis;

    output.d[axis] = exprBuilder.constant(N);

    for (int i = axis + 1; i < inputs[0].nbDims; i++) {
        output.d[i] = exprBuilder.constant(1);
    }

    return output;
}

ILayer* InnerProductTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

const char* InnerProductPluginCreator::getPluginName() const {
    return "InnerProduct";
}

// REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);

}  //  namespace TNN_NS
