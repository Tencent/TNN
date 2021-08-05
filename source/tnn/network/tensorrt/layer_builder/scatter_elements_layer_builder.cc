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
// COElementsITIONS OF ANY KIElements, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(ScatterElements, LAYER_SCATTERELEMENTS);

bool ScatterElementsTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    return (inOut[pos].type == nvinfer1::DataType::kINT32 || inOut[pos].type == nvinfer1::DataType::kFLOAT);
}

Status ScatterElementsTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* ScatterElementsTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "ScatterElements";
}

nvinfer1::DataType ScatterElementsTRTPluginLayerBuilder::getOutputDataType(int index,
        const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

ILayer* ScatterElementsTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs ScatterElementsTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    DimsExprs output(inputs[0]);
    auto resource = dynamic_cast<ScatterElementsLayerResource *>(resource_);

    if (nbInputs < 3) {
        auto data_dims = resource->data.GetBufferDims();
        output.nbDims = data_dims.size();
        for (int i = 0; i < data_dims.size(); i++) {
            output.d[i] = exprBuilder.constant(data_dims[i]);
        }
    }
    return output;
}

const char* ScatterElementsPluginCreator::getPluginName() const noexcept {
    return "ScatterElements";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(ScatterElements, LAYER_SCATTER_ELEMENTS);

}  //  namespace TNN_NS
