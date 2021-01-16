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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Reshape, LAYER_RESHAPE);

bool ReshapeTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW
        && inOut[pos].type == inOut[0].type);
}

const char* ReshapeTRTPluginLayerBuilder::getPluginType() const {
    return "Reshape";
}

nvinfer1::DataType ReshapeTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* ReshapeTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs ReshapeTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    auto param = dynamic_cast<ReshapeLayerParam*>(param_);
    DimsExprs output;
    output.nbDims = param->shape.size() + param->axis;
    for (int i = 0; i < param->axis; i++) {
        output.d[i] = inputs[0].d[i];
    }
    int infer_dim_count = 0;
    int infer_dim_pos = -1;
    for (int i = param->axis, j = 0; i < param->num_axes; i++, j++) {
        if (param->shape[j] == -1) {
            infer_dim_count += 1;
            infer_dim_pos  = i;
            output.d[i] = exprBuilder.constant(1);
        } else if (param->shape[j] == 0) {
            output.d[i] = inputs[0].d[i];
        } else {
            output.d[i] = exprBuilder.constant(param->shape[j]);
        }
    }
    if (infer_dim_count == 0 && infer_dim_pos == -1) return output;

    auto in_cnt = exprBuilder.constant(1);
    for (int i = 0; i < inputs[0].nbDims; i++) {
        in_cnt = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[i], *in_cnt);
    }

    auto out_cnt = exprBuilder.constant(1);
    for (int i = 0; i < output.nbDims; i++) {
        out_cnt = exprBuilder.operation(DimensionOperation::kPROD, *output.d[i], *out_cnt);
    }

    auto infer_dim_v = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *in_cnt, *out_cnt);
    output.d[infer_dim_pos] = infer_dim_v;
    return output;
}

const char* ReshapePluginCreator::getPluginName() const {
    return "Reshape";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Reshape, LAYER_RESHAPE);


}  //  namespace TNN_NS
