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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Einsum, LAYER_EINSUM);

bool EinsumTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW
        && inOut[pos].type == inOut[0].type);
}

Status EinsumTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* EinsumTRTPluginLayerBuilder::getPluginType() const {
    return "Einsum";
}

nvinfer1::DataType EinsumTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* EinsumTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs EinsumTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    auto param = dynamic_cast<EinsumLayerParam *>(param_);
    DimsExprs output;
    auto output_dims = output_blobs_[0]->GetBlobDesc().dims;
    output.nbDims = output_dims.size();
    for (int i = 1; i < output.nbDims; i++) {
        output.d[i] = exprBuilder.constant(output_dims[i]);
    }
    output.d[0] = inputs[0].d[0];
    return output;
}

const char* EinsumPluginCreator::getPluginName() const {
    return "Einsum";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Einsum, LAYER_EINSUM);

}  //  namespace TNN_NS

