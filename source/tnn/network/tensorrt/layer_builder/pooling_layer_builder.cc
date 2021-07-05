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

#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Pooling, LAYER_POOLING);

bool PoolingTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kNCHW);
}

Status PoolingTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* PoolingTRTPluginLayerBuilder::getPluginType() const {
    return "Pooling";
}

nvinfer1::DataType PoolingTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* PoolingTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<PoolingLayerParam*>(param_);
    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    bool int8 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetInt8Mode();

    bool symmetric = (paramlist->pads[0] == paramlist->pads[1]) && (paramlist->pads[2] == paramlist->pads[3]);
    if (symmetric && (paramlist->is_global_pool || (int8 && paramlist->pool_type == 1) || paramlist->is_adaptive_pool)) {
        return TensorRTPluginLayerBuilder::AddToNetwork(network);
    }

    Dims kernelSize(ConvertToTRTDimsReverse(paramlist->kernels));
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();

    PoolingType type;
    if (paramlist->pool_type == 0) {
        type = PoolingType::kMAX;
    } else {
        type = PoolingType::kAVERAGE;
    }

    IPoolingLayer *layer;
    auto pads = paramlist->pads;

    bool padNeg = false;
    for(const auto& p : pads)
        padNeg |= p < 0;

    if (padNeg) {
        DimsVector postPadding{pads[3], pads[1]};
        DimsVector  prePadding{pads[2], pads[0]};
        IPaddingLayer* padding_layer = network->addPaddingNd(*input_tensor,
                                                    ConvertToTRTDims(prePadding),
                                                    ConvertToTRTDims(postPadding));
        input_tensor = padding_layer->getOutput(0);
        pads = {0, 0, 0, 0};
    }
    layer = network->addPoolingNd(*input_tensor, type, kernelSize);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
        layer->setStrideNd(ConvertToTRTDimsReverse(paramlist->strides));
        if (!padNeg) {
            if (symmetric) {
                layer->setPaddingNd(ConvertPaddingToTRTDims(pads));
            } else {
                DimsVector postPadding{pads[3], pads[1]};
                DimsVector  prePadding{pads[2], pads[0]};
                layer->setPrePadding(ConvertToTRTDims(prePadding));
                layer->setPostPadding(ConvertToTRTDims(postPadding));
            }
        }
        if (paramlist->pad_type == -1) {
            if (paramlist->ceil_mode == 1) {
                layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_UP);
            } else {
                layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_DOWN);
            }
        } else if (paramlist->pad_type == 0) {
            layer->setPaddingMode(PaddingMode::kSAME_UPPER);
        } else if (paramlist->pad_type == 1) {
            layer->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);
        }
        if (paramlist->pool_type == 1) {
            layer->setAverageCountExcludesPadding(true);
        }
    }
    if (int8 && std::dynamic_pointer_cast<TensorRTTensor>(output_foreign_tensor)->GetInt8Mode()) {
        float output_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(
            output_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        return AddInt8OutputQDQLayers(network, layer->getOutput(0), output_foreign_tensor, output_scale_value, 1 / output_scale_value);
    }
    return layer;
}

DimsExprs PoolingTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputDims, nvinfer1::IExprBuilder& exprBuilder) {
    auto paramlist = dynamic_cast<PoolingLayerParam*>(param_);
    if (paramlist->is_adaptive_pool) {
        DimsExprs output(inputs[0]);
        output.d[2] = exprBuilder.constant(paramlist->output_shape[1]);
        output.d[3] = exprBuilder.constant(paramlist->output_shape[0]);
        return output;
    } else if (paramlist->is_global_pool) {
        DimsExprs output(inputs[0]);
        output.d[2] = exprBuilder.constant(1);
        output.d[3] = exprBuilder.constant(1);
        return output;
    }

    return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputDims, exprBuilder);
}

const char* PoolingPluginCreator::getPluginName() const {
    return "Pooling";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Pooling, LAYER_POOLING);
REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Pooling, LAYER_POOLING_3D);

}  //  namespace TNN_NS
