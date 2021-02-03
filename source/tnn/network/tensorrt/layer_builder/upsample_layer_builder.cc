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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Upsample, LAYER_UPSAMPLE);

bool UpsampleTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kINT32);
}

const char* UpsampleTRTPluginLayerBuilder::getPluginType() const {
    return "Upsample";
}

nvinfer1::DataType UpsampleTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* UpsampleTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs UpsampleTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    UpsampleLayerParam* param = dynamic_cast<UpsampleLayerParam *>(param_);
    DimsExprs output(inputs[0]);
    auto scales = param->scales;
    auto sizes = param->dims;
    if (sizes.size() <= 0) {
        if (param->mode == 1 || param->mode == 2 || param->mode == 3) {
            auto scale_0 = exprBuilder.constant(scales[0]);
            auto scale_1 = exprBuilder.constant(scales[1]);
            output.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *scale_0);
            output.d[3] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[3], *scale_1);
        }
    } else {
        output.d[2] = exprBuilder.constant(sizes[1]);
        output.d[3] = exprBuilder.constant(sizes[0]);
    }
    return output;
}

const char* UpsamplePluginCreator::getPluginName() const {
    return "Upsample";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Upsample, LAYER_UPSAMPLE);

}  //  namespace TNN_NS
