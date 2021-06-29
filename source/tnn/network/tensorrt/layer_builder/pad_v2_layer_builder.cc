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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(PadV2, LAYER_PADV2);

static bool UseTRTPaddingND(PadLayerParam* paramlist) {
    // Only zero-padding is supported.
    if (paramlist->type != 0 || paramlist->value != 0) {
        return false;
    }

    // input must have 4 dimensions or more.
    if (paramlist->pads.size() < 8) {
        return false;
    }

    // The padding can only be applied along the two innermost dimensions.
    int dim_size = paramlist->pads.size() / 2; 
    for (int i = 0; i < dim_size - 2; ++i) {
        if (paramlist->pads[i] != 0 || paramlist->pads[i + dim_size] != 0) {
            return false;
        }
    }

    return true;
}

bool PadV2TRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    bool base_check = (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kINT32
           || inOut[pos].type == nvinfer1::DataType::kHALF) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW;
    if(nbInputs == 1) {
        return base_check && inOut[pos].type == inOut[0].type;
    } else if(pos == 1) {
        return base_check && inOut[pos].type == nvinfer1::DataType::kINT32;
    } else {
        return base_check && inOut[pos].type == inOut[0].type;
    }
}

Status PadV2TRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* PadV2TRTPluginLayerBuilder::getPluginType() const {
    return "PadV2";
}

nvinfer1::DataType PadV2TRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* PadV2TRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<PadLayerParam*>(param_);

    if (!UseTRTPaddingND(paramlist)) {
        return TensorRTPluginLayerBuilder::AddToNetwork(network);
    }

    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();
    std::vector<int> pads = paramlist->pads;
    int dim_size = pads.size() / 2;
    // use IPaddingLayer
    IPaddingLayer* pad_layer;
    Dims pre_padding = ConvertToTRTDims({pads[dim_size - 2], pads[dim_size - 1]});
    Dims post_padding = ConvertToTRTDims({pads[2 * dim_size - 2], pads[2 * dim_size - 1]});
    pad_layer = network->addPaddingNd(*input_tensor, pre_padding, post_padding);

    return pad_layer;
}

DimsExprs PadV2TRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInput, nvinfer1::IExprBuilder& exprBuilder) {
    DimsExprs output(inputs[0]);
    auto param = dynamic_cast<PadLayerParam*>(param_);
    auto output_dims = input_blobs_[0]->GetBlobDesc().dims;
    int dim_size = param->pads.size()/2;
    dim_size = dim_size <= output_dims.size() ? dim_size : output_dims.size();
    for (int i = 0; i < dim_size; ++i) {
        auto sum = exprBuilder.constant(param->pads[i] + param->pads[dim_size + i]);
        output.d[i] = exprBuilder.operation(DimensionOperation::kSUM, *output.d[i], *sum);
    }
    return output;
}

const char* PadV2PluginCreator::getPluginName() const {
    return "PadV2";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(PadV2, LAYER_PADV2);

}  //  namespace TNN_NS
