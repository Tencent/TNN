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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(CbamFusedReduce, LAYER_CBAM_FUSED_REDUCE);

bool CbamFusedReduceTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs &&
        inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
        (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF);
}

const char* CbamFusedReduceTRTPluginLayerBuilder::getPluginType() const {
    return "CbamFusedReduce";
}

nvinfer1::DataType CbamFusedReduceTRTPluginLayerBuilder::getOutputDataType(int index,
        const nvinfer1::DataType* inputTypes, int nbInputs) const {
    return inputTypes[0];
}

ILayer* CbamFusedReduceTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();

    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

const char* CbamFusedReducePluginCreator::getPluginName() const {
    return "CbamFusedReduce";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(CbamFusedReduce, LAYER_CBAM_FUSED_REDUCE);

}  //  namespace TNN_NS
