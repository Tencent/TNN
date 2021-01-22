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

#include "tnn/network/tensorrt/dimension_expr.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Convolution3D, LAYER_CONVOLUTION_3D);

bool Convolution3DTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kNCHW;
}

const char* Convolution3DTRTPluginLayerBuilder::getPluginType() const {
    return "Convolution3D";
}

nvinfer1::DataType Convolution3DTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

DimsExprs Convolution3DTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {

    ConvLayerParam* conv_param = dynamic_cast<ConvLayerParam*>(param_);
    if (!conv_param) {
        LOGE("Convolution3DTRTPluginLayerBuilder got null param\n");
        return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
    }

    if (inputs[0].nbDims <5) {
        LOGE("Convolution3DTRTPluginLayerBuilder expect dims.size() >= 5\n");
        return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
    }

    for(int i=0;i<3;i++) {
        if (conv_param->pads[i*2] != conv_param->pads[i*2+1]) {
            LOGE("Convolution3DTRTPluginLayerBuilder does not support asymmetric padding.\n");
            return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
        }
    }

    nvinfer1::IExprBuilder& e = exprBuilder;


    const int pad_w_begin = conv_param->pads[0];
    const int pad_h_begin = conv_param->pads[2];
    const int pad_d_begin = conv_param->pads[4];

    const int kernel_w = conv_param->kernels[0];
    const int kernel_h = conv_param->kernels[1];
    const int kernel_d = conv_param->kernels[2];

    const int stride_w = conv_param->strides[0];
    const int stride_h = conv_param->strides[1];
    const int stride_d = conv_param->strides[2];

    const int dilation_w = conv_param->dialations[0];
    const int dilation_h = conv_param->dialations[1];
    const int dilation_d = conv_param->dialations[2];

    DimensionExpr height_out(nullptr, e);
    DimensionExpr width_out(nullptr, e);
    DimensionExpr depth_out(nullptr, e);

    DimensionExpr depth(inputs[0].d[2], e);
    DimensionExpr height(inputs[0].d[3], e);
    DimensionExpr width(inputs[0].d[4], e);

    const int pad_type = conv_param->pad_type;
    if (pad_type == -1)  // default padding following the proto setting
    {
        int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
        int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
        int kernel_extent_d = dilation_d * (kernel_d - 1) + 1;

        height_out = (height + 2 * pad_h_begin - kernel_extent_h) / stride_h + 1;
        width_out  = (width + 2 * pad_w_begin - kernel_extent_w) / stride_w + 1;
        depth_out  = (depth + 2 * pad_d_begin - kernel_extent_d) / stride_d + 1;

    } else if (pad_type == 0) {// SAME type
        height_out = ceil_div(height, stride_h);
        width_out  = ceil_div(width , stride_w);
        depth_out  = ceil_div(depth , stride_d);

    } else if (pad_type == 1)  {// VALID type
        height_out = ceil_div(height - kernel_h + 1, stride_h);
        width_out  = ceil_div(width  - kernel_w + 1, stride_w);
        depth_out  = ceil_div(depth  - kernel_d + 1, stride_d);

    } else {
        LOGE("Convolution3DTRTPluginLayerBuilder only support default padding m\n");
        return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
    }

    DimsExprs output(inputs[0]);

    output.d[1] = e.constant(conv_param->output_channel);
    output.d[2] = depth_out.expr();
    output.d[3] = height_out.expr();
    output.d[4] = width_out.expr();

    return output;
}

ILayer* Convolution3DTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

const char* Convolution3DPluginCreator::getPluginName() const {
    return "Convolution3D";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Convolution3D, LAYER_CONVOLUTION_3D);

}  //  namespace TNN_NS
