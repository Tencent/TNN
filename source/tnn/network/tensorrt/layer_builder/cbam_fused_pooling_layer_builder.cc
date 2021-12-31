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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(CbamFusedPooling, LAYER_CBAM_FUSED_POOLING);

bool CbamFusedPoolingTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    return nbInputs == 1 && nbOutputs == 2 && pos < nbInputs + nbOutputs &&
        inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
        (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) && (pos == 0 || inOut[pos].type == inOut[0].type);
}

Status CbamFusedPoolingTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* CbamFusedPoolingTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "CbamFusedPooling";
}

nvinfer1::DataType CbamFusedPoolingTRTPluginLayerBuilder::getOutputDataType(int index,
        const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* CbamFusedPoolingTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();

    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs CbamFusedPoolingTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    DimsExprs output(inputs[0]);
    output.d[2] = exprBuilder.constant(1);
    output.d[3] = exprBuilder.constant(1);
    return output;
}

const char* CbamFusedPoolingPluginCreator::getPluginName() const noexcept {
    return "CbamFusedPooling";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(CbamFusedPooling, LAYER_CBAM_FUSED_POOLING);

}  //  namespace TNN_NS
