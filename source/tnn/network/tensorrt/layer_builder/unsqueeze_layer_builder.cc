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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Unsqueeze, LAYER_UNSQUEEZE);

bool UnsqueezeTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF ||
        inOut[pos].type == nvinfer1::DataType::kINT32) &&
        inOut[pos].format == nvinfer1::TensorFormat::kNCHW &&
        inOut[pos].type == inOut[0].type;
}

Status UnsqueezeTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* UnsqueezeTRTPluginLayerBuilder::getPluginType() const {
    return "Unsqueeze";
}

nvinfer1::DataType UnsqueezeTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* UnsqueezeTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    if (GetInputITensors()[0]->getDimensions().nbDims == 0) {
        auto param = dynamic_cast<UnsqueezeLayerParam*>(param_);
        nvinfer1::Dims trt_dims;
        trt_dims.nbDims = 1;
        trt_dims.d[0] = 1;
        nvinfer1::Weights const_weight;
        const_weight.type = nvinfer1::DataType::kINT32;
        const_weight.values = &param->axes[0];
        const_weight.count = 1;
        ILayer* constant_layer = network->addConstant(trt_dims, const_weight);
        IShuffleLayer* shuffle_layer = network->addShuffle(*GetInputITensors()[0]);
        nvinfer1::Dims d;
        d.nbDims = 1;
        d.d[0] = 1;
        shuffle_layer->setReshapeDimensions(d);
        return shuffle_layer;
    }
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs UnsqueezeTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    auto param = dynamic_cast<UnsqueezeLayerParam*>(param_);
    auto axes = param->axes;
    DimsExprs output;
    output.nbDims = axes.size() + inputs[0].nbDims;
    for (auto& axis : axes) {
        axis = axis < 0 ? axis + inputs[0].nbDims + 1 : axis;
    }

    int in_index = 0;
    for (int i = 0; i < output.nbDims; i++) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            output.d[i] = inputs[0].d[in_index++];
        } else {
            output.d[i] = exprBuilder.constant(1);
        }
    }
    return output;
}

const char* UnsqueezePluginCreator::getPluginName() const {
    return "Unsqueeze";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Unsqueeze, LAYER_UNSQUEEZE);


}  //  namespace TNN_NS
