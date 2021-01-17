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

const IDimensionExpr * add(const IDimensionExpr * a, const IDimensionExpr * b, nvinfer1::IExprBuilder& exprBuilder) {
    return exprBuilder.operation(DimensionOperation::kSUM, *a, *b);
}

const IDimensionExpr * sub(const IDimensionExpr * a, const IDimensionExpr * b, nvinfer1::IExprBuilder& exprBuilder) {
    return exprBuilder.operation(DimensionOperation::kSUB, *a, *b);
}

const IDimensionExpr * mul(const IDimensionExpr * a, const IDimensionExpr * b, nvinfer1::IExprBuilder& exprBuilder) {
    return exprBuilder.operation(DimensionOperation::kPROD, *a, *b);
}

const IDimensionExpr * floor_div(const IDimensionExpr * a, const IDimensionExpr * b, nvinfer1::IExprBuilder& exprBuilder) {
    return exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *a, *b);
}

const IDimensionExpr * ceil_div(const IDimensionExpr * a, const IDimensionExpr * b, nvinfer1::IExprBuilder& exprBuilder) {
    return exprBuilder.operation(DimensionOperation::kCEIL_DIV, *a, *b);
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

    const IDimensionExpr * pad_w = e.constant(conv_param->pads[0]);
    const IDimensionExpr * pad_h = e.constant(conv_param->pads[2]);
    const IDimensionExpr * pad_d = e.constant(conv_param->pads[4]);

    const IDimensionExpr * kernel_w = e.constant(conv_param->kernels[0]);
    const IDimensionExpr * kernel_h = e.constant(conv_param->kernels[1]);
    const IDimensionExpr * kernel_d = e.constant(conv_param->kernels[2]);

    const IDimensionExpr * stride_w = e.constant(conv_param->strides[0]);
    const IDimensionExpr * stride_h = e.constant(conv_param->strides[1]);
    const IDimensionExpr * stride_d = e.constant(conv_param->strides[2]);

    const IDimensionExpr * dilation_w = e.constant(conv_param->dialations[0]);
    const IDimensionExpr * dilation_h = e.constant(conv_param->dialations[1]);
    const IDimensionExpr * dilation_d = e.constant(conv_param->dialations[2]);

    const IDimensionExpr * one = e.constant(1);
    const IDimensionExpr * two = e.constant(2);


    const IDimensionExpr * depth  = inputs[0].d[2];
    const IDimensionExpr * height = inputs[0].d[3];
    const IDimensionExpr * width  = inputs[0].d[4];

    const IDimensionExpr * height_out;
    const IDimensionExpr * width_out;
    const IDimensionExpr * depth_out;

    const int pad_type = conv_param->pad_type;
    if (pad_type == -1)  // default padding following the proto setting
    {
        // kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
        // kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
        // kernel_extent_d = dilation_d * (kernel_d - 1) + 1;
        const IDimensionExpr * kernel_extent_w = add(mul(dilation_w, sub(kernel_w, one, e), e), one, e);
        const IDimensionExpr * kernel_extent_h = add(mul(dilation_h, sub(kernel_h, one, e), e), one, e);
        const IDimensionExpr * kernel_extent_d = add(mul(dilation_d, sub(kernel_d, one, e), e), one, e);

        // height_out = (height + 2 * pad_h_begin - kernel_extent_h) / stride_h + 1;
        // width_out  = (width + 2 * pad_w_begin - kernel_extent_w) / stride_w + 1;
        // depth_out  = (depth + 2 * pad_d_begin - kernel_extent_d) / stride_d + 1;
        auto default_padding_expr = [&](const IDimensionExpr * whd,    const IDimensionExpr * pad, 
                                        const IDimensionExpr * kernelext, const IDimensionExpr * stride)
                {
                    auto expand_dim = sub(add(whd, mul(pad, two, e), e), kernelext, e);
                    return add(floor_div(expand_dim, stride, e), one, e);
                };

        height_out = default_padding_expr(height, pad_h, kernel_extent_h, stride_h);
        width_out  = default_padding_expr(width , pad_w, kernel_extent_w, stride_w);
        depth_out  = default_padding_expr(depth , pad_d, kernel_extent_d, stride_d);

    } else if (pad_type == 0) {// SAME type
        // height_out = static_cast<int>(std::ceil(float(height) / float(stride_h)));
        // width_out  = static_cast<int>(std::ceil(float(width) / float(stride_w)));
        // depth_out  = static_cast<int>(std::ceil(float(depth) / float(stride_d)));
        auto same_padding_expr = [&](const IDimensionExpr * whd, const IDimensionExpr *stride) 
                {
                    return ceil_div(whd, stride, e);
                };
        height_out = same_padding_expr(height, stride_h);
        width_out  = same_padding_expr(width , stride_w);
        depth_out  = same_padding_expr(depth , stride_d);

    } else if (pad_type == 1)  {// VALID type
        // height_out = static_cast<int>(std::ceil(float(height - kernel_h + 1) / float(stride_h)));
        // width_out  = static_cast<int>(std::ceil(float(width - kernel_w + 1) / float(stride_w)));
        // depth_out  = static_cast<int>(std::ceil(float(depth - kernel_d + 1) / float(stride_d)));
        auto valid_padding_expr = [&](const IDimensionExpr * whd, const IDimensionExpr *stride, 
                                      const IDimensionExpr * kernel) 
                {
                    return ceil_div(add(sub(whd, kernel, e), one, e), stride, e);
                };

        height_out = valid_padding_expr(height, stride_h, kernel_h);
        width_out  = valid_padding_expr(width , stride_w, kernel_w);
        depth_out  = valid_padding_expr(depth , stride_d, kernel_d);
    } else {
        LOGE("Convolution3DTRTPluginLayerBuilder only support default padding m\n");
        return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
    }

    DimsExprs output(inputs[0]);

    output.d[1] = e.constant(conv_param->output_channel);
    output.d[2] = depth_out;
    output.d[3] = height_out;
    output.d[4] = width_out;

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
