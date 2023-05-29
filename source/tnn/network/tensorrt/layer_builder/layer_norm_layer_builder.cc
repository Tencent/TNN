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
#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(LayerNorm, LAYER_LAYER_NORM);

bool LayerNormTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    bool layout_check = inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;

    bool datatype_check = true;
    if (pos == 0) {
        datatype_check = inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF;
    } else {
        datatype_check = inOut[pos].type == inOut[0].type;
    }

    return layout_check && datatype_check;
}

Status LayerNormTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* LayerNormTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "LayerNorm";
}

nvinfer1::DataType LayerNormTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* LayerNormTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 86
    auto layer_param = dynamic_cast<LayerNormLayerParam*>(param_);
    if (!layer_param) {
        LOGE("LayerNormTRTPluginLayerBuilder: Unable to get layer param.");
        return nullptr;
    }

    float epsilon = layer_param->eps;

    std::vector<ITensor*> input_tensors;
    for (int i = 0; i < input_blobs_.size(); i++) {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[i])->GetForeignTensor();
        input_tensors.push_back(std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor());
    }
    auto* input = input_tensors[0];
    auto* scale = input_tensors[1];
    auto* bias = input_tensors[2];
    int axis = input->getDimensions().nbDims - layer_param->reduce_dims_size;
    uint32_t axesMask{0};

    // Populate axesMask with axis values
    for (int32_t i = axis; i < input->getDimensions().nbDims; i++)
    {
        axesMask |= 1 << i;
    }

    // Broadcast scale and bias to input size
    BroadcastTensors(network, input, scale);
    BroadcastTensors(network, input, bias);

    auto* layer = network->addNormalization(*input, *scale, *bias, axesMask);
    layer->setEpsilon(epsilon);
    return layer;
#else
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
#endif
}

DimsExprs LayerNormTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
}

const char* LayerNormPluginCreator::getPluginName() const noexcept {
    return "LayerNorm";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(LayerNorm, LAYER_LAYER_NORM);

}  //  namespace TNN_NS
