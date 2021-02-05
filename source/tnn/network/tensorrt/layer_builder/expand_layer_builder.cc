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

#include <cudnn.h>

#include "tnn/interpreter/layer_param.h"
#include "tnn/network/tensorrt/dimension_expr.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Expand, LAYER_EXPAND);

bool ExpandTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return (((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW
        && inOut[pos].type == inOut[0].type) || inOut[pos].type == nvinfer1::DataType::kINT32);
}

const char* ExpandTRTPluginLayerBuilder::getPluginType() const {
    return "Expand";
}

nvinfer1::DataType ExpandTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}


DimsExprs ExpandTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {

    nvinfer1::IExprBuilder& e = exprBuilder;
   
    auto layer_param = dynamic_cast<ExpandLayerParam*>(param_);
    if (!layer_param) {
        LOGE("ExpandTRTPluginLayerBuilder::getOutputDimensions got null param\n");
        return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
    }

    // TODO, deal with the case: shape_dims from TensorData
    auto shape_dims = layer_param->shape;

    DimsExprs expand_dims;
    expand_dims.nbDims = shape_dims.size();
    for(int i=0;i<shape_dims.size();i++) {
        expand_dims.d[i] = DimensionExpr(shape_dims[i], e).expr();
    }

    int nb_dims_diff = std::abs(expand_dims.nbDims - inputs[0].nbDims);

    DimsExprs output;
    if (expand_dims.nbDims > inputs[0].nbDims) {
        output = expand_dims;
        for(int i=0;i<output.nbDims;i++) {
            if (i >= nb_dims_diff) {
                output.d[i] = e.operation(nvinfer1::DimensionOperation::kMAX, *expand_dims.d[i], *inputs[0].d[i - nb_dims_diff]);
            }
        }
    } else {
        output = inputs[0];
        for(int i=0;i<output.nbDims;i++) {
            if (i >= nb_dims_diff) {
                output.d[i] = e.operation(nvinfer1::DimensionOperation::kMAX, *expand_dims.d[i - nb_dims_diff], *inputs[0].d[i]);
            }
        }
    }

    return output;
}

ILayer* ExpandTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

const char* ExpandPluginCreator::getPluginName() const {
    return "Expand";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Expand, LAYER_EXPAND);

}  //  namespace TNN_NS
