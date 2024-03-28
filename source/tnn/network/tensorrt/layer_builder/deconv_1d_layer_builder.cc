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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Deconvolution1D, LAYER_DECONVOLUTION_1D);

bool Deconvolution1DTRTPluginLayerBuilder::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                                                   int nbInputs, int nbOutputs) noexcept {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) &&
           inOut[pos].type == inOut[0].type && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

Status Deconvolution1DTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* Deconvolution1DTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "Deconvolution";
}

nvinfer1::DataType Deconvolution1DTRTPluginLayerBuilder::getOutputDataType(int index,
                                                                         const nvinfer1::DataType* inputTypes,
                                                                         int nbInputs) const noexcept {
    return inputTypes[0];
}

DimsExprs Deconvolution1DTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
                                                                  int nbInputs,
                                                                  nvinfer1::IExprBuilder& exprBuilder) noexcept {
    ConvLayerParam* conv_param = dynamic_cast<ConvLayerParam*>(param_);
    if (!conv_param) {
        LOGE("Deconvolution1DTRTPluginLayerBuilder got null param\n");
        return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
    }

    for (int i = 0; i < 1; i++) {
        if (conv_param->pads[i * 2] != conv_param->pads[i * 2 + 1]) {
            LOGE("Deconvolution1DTRTPluginLayerBuilder does not support asymmetric padding.\n");
            return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
        }
    }

    nvinfer1::IExprBuilder& e = exprBuilder;

    const int pad_w_begin = conv_param->pads[0];
    const int kernel_w = conv_param->kernels[0];
    const int stride_w = conv_param->strides[0];
    const int dilation_w = conv_param->dialations[0];

    DimensionExpr width_out(nullptr, &e);
    DimensionExpr width(inputs[0].d[2], &e);

    const int pad_type = conv_param->pad_type;
    int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    if (pad_type == -1) {
        // default padding following the proto setting
        width_out  = stride_w * (width - 1) + kernel_extent_w - 2 * pad_w_begin;
    } else if (pad_type == 0 || pad_type == 3) {
        // SAME type
        width_out  = width * stride_w;
    } else if (pad_type == 1) {
        // VALID type
        width_out  = width * stride_w + std::max(kernel_extent_w - stride_w, 0);
    } else if (pad_type == 2) {
        // FULL type
        width_out  = width * stride_w - (stride_w + kernel_extent_w - 2);
    } else {
        LOGE("Deconvolution1DTRTPluginLayerBuilder only support default padding m\n");
        return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
    }

    DimsExprs output(inputs[0]);

    output.d[1] = e.constant(conv_param->output_channel);
    output.d[2] = width_out.expr();

    return output;
}

const char* Deconvolution1DPluginCreator::getPluginName() const noexcept {
    return "Deconvolution1D";
}

ILayer* Deconvolution1DTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    auto paramlist = dynamic_cast<ConvLayerParam*>(param_);

    if (paramlist->pad_type == -1 && paramlist->dialations[0] == 1 &&
        paramlist->input_channel == 1 && paramlist->output_channel == paramlist->group) {
        return TensorRTPluginLayerBuilder::AddToNetwork(network);
    }

    nvinfer1::ITensor* weight_tensor = nullptr;
    if (input_blobs_.size() > 1) {
        auto weight_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[1])->GetForeignTensor();
        weight_tensor              = std::dynamic_pointer_cast<TensorRTTensor>(weight_foreign_tensor)->GetTensor();
        auto dims                  = weight_tensor->getDimensions();
        paramlist->kernels[0]      = dims.d[2];
        if (paramlist->pad_type == 3) {
            paramlist->input_channel  = dims.d[0] / paramlist->group;
            paramlist->output_channel = dims.d[1] * paramlist->group;
        } else {
            paramlist->input_channel  = dims.d[1];
            paramlist->output_channel = dims.d[0];
        }
    }

    if (paramlist->dialations[0] != 1) {
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

    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();

    ILayer* last_layer;
    DimsHW kernelSize(paramlist->kernels[0], 1);

    //DimsVector unsqueeze_dims(input_tensor->getDimensions().nbDims, 0);
    //unsqueeze_dims.push_back(1);
    //ILayer* layer = AddReshapeToNetwork(network, input_tensor, unsqueeze_dims, (layer_name_ + "unsqueeze").c_str());

    const std::vector<int> axes{3};
    ILayer* layer = AddUnSqueezeToNetwork(network, input_tensor, axes, (layer_name_ + "unsqueeze").c_str());

    auto pads = paramlist->pads;
    IDeconvolutionLayer* deconv_layer;
    if (paramlist->pad_type == -1 || (pads[0] == pads[1]) ) {
        deconv_layer = network->addDeconvolution(*(layer->getOutput(0)), paramlist->output_channel,
            kernelSize, kernelWeights, biasWeights);
        if (deconv_layer != nullptr) {
            deconv_layer->setName(layer_name_.c_str());
            deconv_layer->setStride(DimsHW(paramlist->strides[0], 1));
            deconv_layer->setPadding(DimsHW(paramlist->pads[0], 0));
            deconv_layer->setNbGroups(paramlist->group);
            //deconv_layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_DOWN);
        }
    } else {
        DimsVector postPadding{pads[1], 0};
        DimsVector  prePadding{pads[0], 0};
        deconv_layer = network->addDeconvolution(*(layer->getOutput(0)), paramlist->output_channel, kernelSize,
            kernelWeights, biasWeights);
        if(deconv_layer != NULL) {
            deconv_layer->setName(layer_name_.c_str());
            Dims strides;
            strides.nbDims = 2;
            strides.d[0] = paramlist->strides[0];
            strides.d[1] = 1;
            deconv_layer->setStrideNd(strides);
            deconv_layer->setPrePadding(ConvertToTRTDims(prePadding));
            deconv_layer->setPostPadding(ConvertToTRTDims(postPadding));
#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 71
            Dims dialations;
            dialations.nbDims = 2;
            dialations.d[0] = paramlist->dialations[0];
            dialations.d[1] = 1;
            deconv_layer->setDilationNd(dialations);
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

//    unsqueeze_dims.erase(unsqueeze_dims.end()-1);
//    last_layer = AddReshapeToNetwork(network, last_layer->getOutput(0), unsqueeze_dims, (layer_name_ + "squeeze").c_str());

    last_layer = AddSqueezeToNetwork(network, last_layer->getOutput(0), axes, (layer_name_ + "squeeze").c_str());
    return last_layer;
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Deconvolution1D, LAYER_DECONVOLUTION_1D);

}  //  namespace TNN_NS

