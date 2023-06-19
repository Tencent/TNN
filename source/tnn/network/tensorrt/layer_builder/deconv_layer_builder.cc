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

#include "tnn/network/tensorrt/tensorrt_network.h"
#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"
#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"
#include "tnn/network/tensorrt/dimension_expr.h"
#include "tnn/network/tensorrt/utils.h"
#include <NvInfer.h>

namespace TNN_NS {

// DECLARE_TENSORRT_LAYER_BUILDER(Deconvolution, LAYER_DECONVOLUTION);

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Deconvolution, LAYER_DECONVOLUTION);

bool DeconvolutionTRTPluginLayerBuilder::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                                                   int nbInputs, int nbOutputs) noexcept {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) &&
           inOut[pos].type == inOut[0].type && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

Status DeconvolutionTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* DeconvolutionTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "Deconvolution";
}

nvinfer1::DataType DeconvolutionTRTPluginLayerBuilder::getOutputDataType(int index,
                                                                         const nvinfer1::DataType* inputTypes,
                                                                         int nbInputs) const noexcept {
    return inputTypes[0];
}

DimsExprs DeconvolutionTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
                                                                  int nbInputs,
                                                                  nvinfer1::IExprBuilder& exprBuilder) noexcept {
    ConvLayerParam* conv_param = dynamic_cast<ConvLayerParam*>(param_);
    if (!conv_param) {
        LOGE("ConvolutionTRTPluginLayerBuilder got null param\n");
        return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
    }

    for (int i = 0; i < 2; i++) {
        if (conv_param->pads[i * 2] != conv_param->pads[i * 2 + 1]) {
            LOGE("ConvolutionTRTPluginLayerBuilder does not support asymmetric padding.\n");
            return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
        }
    }

    nvinfer1::IExprBuilder& e = exprBuilder;

    const int pad_w_begin = conv_param->pads[0];
    const int pad_h_begin = conv_param->pads[2];

    const int kernel_w = conv_param->kernels[0];
    const int kernel_h = conv_param->kernels[1];

    const int stride_w = conv_param->strides[0];
    const int stride_h = conv_param->strides[1];

    const int dilation_w = conv_param->dialations[0];
    const int dilation_h = conv_param->dialations[1];

    DimensionExpr height_out(nullptr, &e);
    DimensionExpr width_out(nullptr, &e);

    DimensionExpr height(inputs[0].d[2], &e);
    DimensionExpr width(inputs[0].d[3], &e);

    const int pad_type = conv_param->pad_type;

    int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    if (pad_type == -1) {
        // default padding following the proto setting
        height_out = stride_h * (height - 1) + kernel_extent_h - 2 * pad_h_begin;
        width_out  = stride_w * (width - 1) + kernel_extent_w - 2 * pad_w_begin;
    } else if (pad_type == 0 || pad_type == 3) {
        // SAME type
        height_out = height * stride_h;
        width_out  = width * stride_w;
    } else if (pad_type == 1) {
        // VALID type
        height_out = height * stride_h + std::max(kernel_extent_h - stride_h, 0);
        width_out  = width * stride_w + std::max(kernel_extent_w - stride_w, 0);
    } else if (pad_type == 2) {
        // FULL type
        height_out = height * stride_h - (stride_h + kernel_extent_h - 2);
        width_out  = width * stride_w - (stride_w + kernel_extent_w - 2);
    } else {
        LOGE("ConvolutionTRTPluginLayerBuilder only support default padding m\n");
        return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
    }

    DimsExprs output(inputs[0]);

    output.d[1] = e.constant(conv_param->output_channel);
    output.d[2] = height_out.expr();
    output.d[3] = width_out.expr();

    return output;
}

const char* DeconvolutionPluginCreator::getPluginName() const noexcept {
    return "Deconvolution";
}

ILayer* DeconvolutionTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    auto paramlist = dynamic_cast<ConvLayerParam*>(param_);

    if (paramlist->pad_type == -1 && paramlist->dialations[0] == 1 && paramlist->dialations[1] == 1 &&
        paramlist->input_channel == 1 && paramlist->output_channel == paramlist->group) {
        return TensorRTPluginLayerBuilder::AddToNetwork(network);
    }

    nvinfer1::ITensor* weight_tensor = nullptr;
    if (input_blobs_.size() > 1) {
        auto weight_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[1])->GetForeignTensor();
        weight_tensor              = std::dynamic_pointer_cast<TensorRTTensor>(weight_foreign_tensor)->GetTensor();
        auto dims                  = weight_tensor->getDimensions();
        paramlist->kernels[0]      = dims.d[3];
        paramlist->kernels[1]      = dims.d[2];
        if (paramlist->pad_type == 3) {
            paramlist->input_channel  = dims.d[0] / paramlist->group;
            paramlist->output_channel = dims.d[1] * paramlist->group;
        } else {
            paramlist->input_channel  = dims.d[1];
            paramlist->output_channel = dims.d[0];
        }
    }

    if (paramlist->dialations[1] != 1 || paramlist->dialations[0] != 1) {
        LOGE("TRT does not support dilated deconvolutions");
        return nullptr;
    }
    auto resource = dynamic_cast<ConvLayerResource*>(resource_);
    Weights kernelWeights, biasWeights;
    kernelWeights = ConvertToWeights(&(resource->filter_handle));
    if (paramlist->bias) {
        biasWeights = ConvertToWeights(&(resource->bias_handle));
    } else {
        biasWeights = ConvertToWeights(nullptr, true, resource->filter_handle.GetDataType());
    }

    ILayer* last_layer;
    DimsHW kernelSize(paramlist->kernels[1], paramlist->kernels[0]);
    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();

    auto pads = paramlist->pads;
    IDeconvolutionLayer* deconv_layer;
    if (paramlist->pad_type == -1 || (pads[0] == pads[1] && pads[2] == pads[3])) {
        deconv_layer = network->addDeconvolution(*tensor, paramlist->output_channel,
            kernelSize, kernelWeights, biasWeights);
        if (deconv_layer != nullptr) {
            deconv_layer->setName(layer_name_.c_str());
            deconv_layer->setStride(DimsHW(paramlist->strides[1], paramlist->strides[0]));
            deconv_layer->setPadding(DimsHW(paramlist->pads[2], paramlist->pads[0]));
            deconv_layer->setNbGroups(paramlist->group);
            //deconv_layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_DOWN);
        }
    } else {
        DimsVector postPadding{pads[3], pads[1]};
        DimsVector  prePadding{pads[2], pads[0]};
        deconv_layer = network->addDeconvolution(*tensor, paramlist->output_channel, kernelSize,
            kernelWeights, biasWeights);
        if (deconv_layer != NULL) {
            deconv_layer->setName(layer_name_.c_str());
            deconv_layer->setStrideNd(ConvertToTRTDimsReverse(paramlist->strides));
            deconv_layer->setPrePadding(ConvertToTRTDims(prePadding));
            deconv_layer->setPostPadding(ConvertToTRTDims(postPadding));
#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 71
            deconv_layer->setDilationNd(ConvertToTRTDimsReverse(paramlist->dialations));
#endif
            deconv_layer->setNbGroups(paramlist->group);
        }
    }
    
    if (input_blobs_.size() > 1) {
        deconv_layer->setInput(1, *weight_tensor);
    }
    last_layer = deconv_layer;

    IActivationLayer* activation_layer;
    if (paramlist->activation_type == ActivationType_ReLU) {
        activation_layer = network->addActivation(*(deconv_layer->getOutput(0)), nvinfer1::ActivationType::kRELU);
        last_layer = activation_layer;
    } else if (paramlist->activation_type == ActivationType_ReLU6) {
        activation_layer = network->addActivation(*(deconv_layer->getOutput(0)), nvinfer1::ActivationType::kCLIP);
        activation_layer->setAlpha(0.f);
        activation_layer->setBeta(6.f);
        last_layer = activation_layer;
    } else if (paramlist->activation_type != ActivationType_None) {
        LOGE("Error: Unsupport reshape type(%d)", paramlist->activation_type);
        return nullptr;
    }

    return last_layer;
}

// REGISTER_TENSORRT_LAYER_BUILDER(Deconvolution, LAYER_DECONVOLUTION);
REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Deconvolution, LAYER_DECONVOLUTION);

}  //  namespace TNN_NS

