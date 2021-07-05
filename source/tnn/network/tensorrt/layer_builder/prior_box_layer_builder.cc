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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(PriorBox, LAYER_PRIOR_BOX);

bool PriorBoxTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::TensorFormat::kNCHW
        && inOut[pos].type == inOut[0].type);
}

Status PriorBoxTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* PriorBoxTRTPluginLayerBuilder::getPluginType() const {
    return "PriorBox";
}

nvinfer1::DataType PriorBoxTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* PriorBoxTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs PriorBoxTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    PriorBoxLayerParam* param = dynamic_cast<PriorBoxLayerParam *>(param_);
    int num_priors = static_cast<int>(param->aspect_ratios.size() * param->min_sizes.size());
    if (!param->max_sizes.empty()) {
        for (int i = 0; i < param->max_sizes.size(); ++i) {
            ASSERT(param->max_sizes[i] > param->min_sizes[i]);
            num_priors++;
        }
    }
    DimsExprs output(inputs[0]);
    output.d[0] = exprBuilder.constant(1);
    output.d[1] = exprBuilder.constant(2);
    auto four = exprBuilder.constant(4 * num_priors);
    output.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);
    output.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *output.d[2], *four);
    output.d[3] = exprBuilder.constant(1);
    return output;
}

const char* PriorBoxPluginCreator::getPluginName() const {
    return "PriorBox";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(PriorBox, LAYER_PRIOR_BOX);

}  //  namespace TNN_NS
