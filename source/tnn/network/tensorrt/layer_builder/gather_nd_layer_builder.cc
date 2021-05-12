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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(GatherND, LAYER_GATHERND);

bool GatherNDTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return inOut[0].type == nvinfer1::DataType::kFLOAT;
}

const char* GatherNDTRTPluginLayerBuilder::getPluginType() const {
    return "GatherND";
}

nvinfer1::DataType GatherNDTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* GatherNDTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs GatherNDTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInput, nvinfer1::IExprBuilder& exprBuilder) {
    DimsExprs output;

    output.nbDims = output_blobs_[0]->GetBlobDesc().dims.size();
    for (int i = 0; i < output.nbDims; i++) {
        output.d[i] = exprBuilder.constant(output_blobs_[0]->GetBlobDesc().dims[i]);
    }
    return output;
}

const char* GatherNDPluginCreator::getPluginName() const {
    return "GatherND";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(GatherND, LAYER_GATHERND);

}  //  namespace TNN_NS
