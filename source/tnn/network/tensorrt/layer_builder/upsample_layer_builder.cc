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

bool UpsampleTRTPluginLayerBuilder::supportsFormatCombination (
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kINT32);
}

Status UpsampleTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* UpsampleTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "Upsample";
}

nvinfer1::DataType UpsampleTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* UpsampleTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    auto paramlist = dynamic_cast<UpsampleLayerParam*>(param_);
    if (!paramlist->align_corners) {
        Blob* output_blob = output_blobs_[0];
        auto output_dims = output_blob->GetBlobDesc().dims;
        auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
        auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
        auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();
        IResizeLayer* layer = network->addResize(*input_tensor);
        if (layer != nullptr) {
            layer->setName(layer_name_.c_str());
            if (input_blobs_.size() == 1) {
                if (!paramlist->dims.empty()) {
                    nvinfer1::Dims4 dims(output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
                    layer->setOutputDimensions(dims);
                } else {
                    float scale[4];
                    scale[0] = 1;
                    scale[1] = 1;
                    scale[2] = paramlist->scales[1];
                    scale[3] = paramlist->scales[0];
                    layer->setScales(scale, 4);
                }
            } else if (input_blobs_.size() == 4) {
                auto input_foreign_tensor2 = dynamic_cast<ForeignBlob*>(input_blobs_[input_blobs_.size()-1])->GetForeignTensor();
                auto input_tensor2 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor2)->GetTensor();
                layer->setInput(1, *input_tensor2);
            } else {
                    float scale[4];
                    scale[0] = 1;
                    scale[1] = 1;
                    scale[2] = paramlist->scales[1];
                    scale[3] = paramlist->scales[0];
                    layer->setScales(scale, 4);
            }
            layer->setResizeMode(paramlist->mode == 1 ? ResizeMode::kNEAREST : ResizeMode::kLINEAR);
            layer->setAlignCorners(paramlist->align_corners);
        }
        return layer;
    }

    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs UpsampleTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
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

const char* UpsamplePluginCreator::getPluginName() const noexcept {
    return "Upsample";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Upsample, LAYER_UPSAMPLE);

}  //  namespace TNN_NS

