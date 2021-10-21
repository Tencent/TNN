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
#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Convolution3D, LAYER_CONVOLUTION_3D);

bool Convolution3DTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

Status Convolution3DTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* Convolution3DTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "Convolution3D";
}

nvinfer1::DataType Convolution3DTRTPluginLayerBuilder::getOutputDataType(int index,
        const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
    return inputTypes[0];
}

DimsExprs Convolution3DTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {

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

    DimensionExpr height_out(nullptr, &e);
    DimensionExpr width_out(nullptr, &e);
    DimensionExpr depth_out(nullptr, &e);

    DimensionExpr depth(inputs[0].d[2], &e);
    DimensionExpr height(inputs[0].d[3], &e);
    DimensionExpr width(inputs[0].d[4], &e);

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

ILayer* Convolution3DTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    auto paramlist = dynamic_cast<ConvLayerParam*>(param_);
    auto resource = dynamic_cast<ConvLayerResource*>(resource_);

    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();
    bool int8 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetInt8Mode();

    Weights kernelWeights;
    Weights biasWeights;
    ILayer* last_layer;
    if (int8) {
        float weight_scale_value = *(resource->scale_handle.force_to<float*>());
        float input_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(
            input_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        std::vector<int> dims;
        dims.push_back(paramlist->output_channel);
        dims.push_back(input_blobs_[0]->GetBlobDesc().dims[1] / paramlist->group);
        dims.push_back(paramlist->kernels[1]);
        dims.push_back(paramlist->kernels[0]);
        last_layer = AddInt8WeightQDQLayers(network, &(resource->filter_handle), kernelWeights,
            paramlist->bias ? &(resource->bias_handle) : nullptr, biasWeights,
            1 / (weight_scale_value / input_scale_value), dims);
    } else {
        kernelWeights = ConvertToWeights(&(resource->filter_handle));
        if (paramlist->bias) {
            biasWeights = ConvertToWeights(&(resource->bias_handle));
        } else {
            biasWeights = ConvertToWeights(nullptr, true, resource->filter_handle.GetDataType());
        }
    }

    Dims kernelSize = ConvertToTRTDimsReverse(paramlist->kernels);
    IConvolutionLayer* conv_layer;
    auto pads = paramlist->pads;
    if (paramlist->pad_type == -1 || (pads[0] == pads[1] && pads[2] == pads[3] && pads[4] == pads[5])) {
        conv_layer = network->addConvolutionNd(*input_tensor, paramlist->output_channel, kernelSize,
            kernelWeights, biasWeights);
        if (int8) conv_layer->setInput(1, *(last_layer->getOutput(0)));
        if (conv_layer != nullptr) {
            conv_layer->setName(layer_name_.c_str());
            conv_layer->setStrideNd(ConvertToTRTDimsReverse(paramlist->strides));
            conv_layer->setDilationNd(ConvertToTRTDimsReverse(paramlist->dialations));
            conv_layer->setPaddingNd(ConvertPaddingToTRTDims(paramlist->pads));
            conv_layer->setNbGroups(paramlist->group);
        }
    } else {
        DimsVector postPadding{pads[3], pads[1]};
        DimsVector  prePadding{pads[2], pads[0]};
        IPaddingLayer* padding_layer = network->addPaddingNd(*input_tensor, 
                                                    ConvertToTRTDims(prePadding), 
                                                    ConvertToTRTDims(postPadding));
        ITensor* pad_tensor = padding_layer->getOutput(0);
        conv_layer = network->addConvolutionNd(*pad_tensor, paramlist->output_channel, kernelSize,
            kernelWeights, biasWeights);
        if (int8) conv_layer->setInput(1, *(last_layer->getOutput(0)));
        if(conv_layer != NULL) {
            conv_layer->setName(layer_name_.c_str());
            conv_layer->setStrideNd(ConvertToTRTDimsReverse(paramlist->strides));
            conv_layer->setDilationNd(ConvertToTRTDimsReverse(paramlist->dialations));
            conv_layer->setNbGroups(paramlist->group);
        }
    }

    last_layer = conv_layer;

    if (int8) {
        conv_layer->setPrecision(nvinfer1::DataType::kINT8);
    }

    IActivationLayer* activation_layer;
    if (paramlist->activation_type == ActivationType_ReLU) {
        activation_layer = network->addActivation(*(conv_layer->getOutput(0)), nvinfer1::ActivationType::kRELU);
        last_layer = activation_layer;
    } else if (paramlist->activation_type == ActivationType_ReLU6) {
        activation_layer = network->addActivation(*(conv_layer->getOutput(0)), nvinfer1::ActivationType::kCLIP);
        activation_layer->setAlpha(0.f);
        activation_layer->setBeta(6.f);
        last_layer = activation_layer;
    } else if (paramlist->activation_type != ActivationType_None) {
        LOGE("Error: Unsupport reshape type(%d)", paramlist->activation_type);
        return nullptr;
    }

    if (int8) {
        float output_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(
            output_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        return AddInt8OutputQDQLayers(network, last_layer->getOutput(0), output_foreign_tensor,
            output_scale_value, 1 / output_scale_value);
    }

    return last_layer;
}

const char* Convolution3DPluginCreator::getPluginName() const noexcept {
    return "Convolution3D";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Convolution3D, LAYER_CONVOLUTION_3D);

}  //  namespace TNN_NS
