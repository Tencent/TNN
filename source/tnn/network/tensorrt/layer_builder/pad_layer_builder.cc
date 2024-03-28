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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Pad, LAYER_PAD);

static bool UseTRTPaddingND(PadLayerParam* paramlist) {
    // Only zero-padding is supported.
    if (paramlist->type != 0 || paramlist->value != 0) {
        return false;
    }

    // A must have three dimensions or more.
    if (paramlist->pads.size() < 6) {
        return false;
    }

    // The padding can only be applied along the two innermost dimensions.
    if (paramlist->pads[4] != 0 || paramlist->pads[5] != 0) {
        return false;
    }

    return true;
}

bool PadTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    int channels = inOut[0].dims.d[1];
    bool is_pad_8 = (channels % 8 == 0);
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) ||
        (inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == nvinfer1::TensorFormat::kHWC8 && is_pad_8)) && (pos == 0 || inOut[pos].type == inOut[0].type);
}

Status PadTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* PadTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "Pad";
}

nvinfer1::DataType PadTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* PadTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    auto paramlist = dynamic_cast<PadLayerParam*>(param_);

    if (!UseTRTPaddingND(paramlist)) {
        return TensorRTPluginLayerBuilder::AddToNetwork(network);
    }

    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();
    std::vector<int> pads = paramlist->pads;

    // IPaddingLayer only support 4d Tensor
    // If Input is CHW (supported by pytorch) instead of NCHW,
    // Unsqueeze input tensor to [1, C, H, W]
    ILayer* layer;
    int input_nbDims = input_tensor->getDimensions().nbDims;
    if (input_nbDims == 3) {
        layer = addUnsqueeze(network, *input_tensor, {0});
        layer->setName((layer_name_+"_chw_to_nchw").c_str());
        input_tensor = layer->getOutput(0);
    }

    // use IPaddingLayer
    Dims pre_padding = ConvertToTRTDims({pads[2], pads[0]});
    Dims post_padding = ConvertToTRTDims({pads[3], pads[1]});
    layer = network->addPaddingNd(*input_tensor, pre_padding, post_padding);
    input_tensor = layer->getOutput(0);

    // If Input is CHW (supported by pytorch) instead of NCHW,
    // Squeeze result tensor back to [C, H, W]
    if (input_nbDims == 3) {
        layer = addSqueeze(network, *input_tensor, {0});
        layer->setName((layer_name_+"_nchw_to_chw").c_str());
    }

    return layer;
}

DimsExprs PadTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInput, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    DimsExprs output(inputs[0]);
    auto param = dynamic_cast<PadLayerParam*>(param_);
    auto pads0 = exprBuilder.constant(param->pads[0] + param->pads[1]);
    auto pads1 = exprBuilder.constant(param->pads[2] + param->pads[3]);
    auto pads2 = exprBuilder.constant(param->pads[4] + param->pads[5]);

    int nbDims = inputs[0].nbDims;
    // Support NCHW(nbDims==4) or CHW(nbDims==3)
    output.d[nbDims-1] = exprBuilder.operation(DimensionOperation::kSUM, *output.d[nbDims-1], *pads0);
    output.d[nbDims-2] = exprBuilder.operation(DimensionOperation::kSUM, *output.d[nbDims-2], *pads1);
    output.d[nbDims-3] = exprBuilder.operation(DimensionOperation::kSUM, *output.d[nbDims-3], *pads2);

    return output;
}

const char* PadPluginCreator::getPluginName() const noexcept {
    return "Pad";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Pad, LAYER_PAD);

}  //  namespace TNN_NS
