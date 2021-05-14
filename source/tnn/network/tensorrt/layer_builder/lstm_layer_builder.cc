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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(LSTMONNX, LAYER_LSTMONNX);

bool LSTMONNXTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kNCHW;
}

const char* LSTMONNXTRTPluginLayerBuilder::getPluginType() const {
    return "LSTMONNX";
}

nvinfer1::DataType LSTMONNXTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}


DimsExprs LSTMONNXTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {

    nvinfer1::IExprBuilder& e = exprBuilder;
   
    auto layer_param = dynamic_cast<LSTMONNXLayerParam*>(param_);
    if (!layer_param) {
        LOGE("LSTMONNXTRTPluginLayerBuilder::getOutputDimensions got null param\n");
        return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
    }

    DimensionExpr num_directions(layer_param->direction >=2 ? 2 : 1, e);
    DimensionExpr output_size(layer_param->hidden_size, e);
    
    DimensionExpr sequence_len(inputs[0].d[0], e);
    DimensionExpr batch(inputs[0].d[1], e);

    DimsExprs output;

    switch(index) {
        case 0:
            output.d[0] = sequence_len.expr();
            output.d[1] = batch.expr();
            output.d[2] = (num_directions * output_size).expr();
            output.nbDims = 3;
            break;
        case 1:
        case 2:
            output.d[0] = num_directions.expr();
            output.d[1] = batch.expr();
            output.d[2] = output_size.expr();
            output.nbDims = 3;
            break;
        default:
            break;
    }

    return output;
}

ILayer* LSTMONNXTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

const char* LSTMONNXPluginCreator::getPluginName() const {
    return "LSTMONNX";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(LSTMONNX, LAYER_LSTMONNX);

}  //  namespace TNN_NS
