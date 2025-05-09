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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(EffectiveTransformer, LAYER_EFFECTIVE_TRANSFORMER);

bool EffectiveTransformerTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    auto layer_param = dynamic_cast<EffectiveTransformerLayerParam*>(param_);
    if (!layer_param) {
        LOGE("EffectiveTransformerTRTPluginLayerBuilder: Unable to get layer param.");
        return false;
    }

    bool layout_check = inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;

    bool datatype_check = true;
    if (pos == 0) {
        datatype_check = inOut[pos].type == nvinfer1::DataType::kHALF;
    } else if (pos == nbInputs || (layer_param->is_remove_padding && pos == 1)) {
        datatype_check = inOut[pos].type == inOut[0].type;
    } else {
        datatype_check = inOut[pos].type == nvinfer1::DataType::kINT32;
    }

    return layout_check && datatype_check;
}

Status EffectiveTransformerTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* EffectiveTransformerTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "EffectiveTransformer";
}

nvinfer1::DataType EffectiveTransformerTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    if (index == 0) {
        return inputTypes[0];
    }
    return nvinfer1::DataType::kINT32;
}

ILayer* EffectiveTransformerTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs EffectiveTransformerTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    auto layer_param = dynamic_cast<EffectiveTransformerLayerParam*>(param_);
    if (!layer_param) {
        LOGE("EffectiveTransformerTRTPluginLayerBuilder: Unable to get layer param.");
        return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
    }

    if (layer_param->is_remove_padding && index == 1) {
        nvinfer1::DimsExprs output;
        output.nbDims = 1;
        output.d[0] = inputs[0].d[0];
        for (int i = 1; i < inputs[0].nbDims - 1; ++i) {
            output.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *(inputs[0].d[i]), *(output.d[0]));
        }
        const nvinfer1::IDimensionExpr *one = exprBuilder.constant(1); // for token_number
        output.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *one, *(output.d[0]));
        return output;
    }
    if (layer_param->is_remove_padding && index == 2) {
        nvinfer1::DimsExprs output;
        output.nbDims = 1;
        output.d[0] = inputs[0].d[0];
        const nvinfer1::IDimensionExpr *one = exprBuilder.constant(1);
        output.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *one, *(output.d[0]));
        return output;
    }
    if (!layer_param->is_remove_padding && index == 1) {
        nvinfer1::DimsExprs output;
        output.nbDims = 1;
        output.d[0] = exprBuilder.constant(1);
        return output;
    }
    return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
}

const char* EffectiveTransformerPluginCreator::getPluginName() const noexcept {
    return "EffectiveTransformer";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(EffectiveTransformer, LAYER_EFFECTIVE_TRANSFORMER);

}  //  namespace TNN_NS
