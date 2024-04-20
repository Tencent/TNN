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
    Blob* output_blob = output_blobs_[0];
    auto output_dims = output_blob->GetBlobDesc().dims;
    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();
    ShapeTensor out_shape_tensor;
    if (input_blobs_.size() == 2) {
        // when got 2 blobs, upsample is converted from torch op, second input is hw shape tensor
        auto input_tensors = GetInputITensors();
        auto input_foreign_tensor2 = dynamic_cast<ForeignBlob*>(input_blobs_[input_blobs_.size()-1])->GetForeignTensor();
        auto input_tensor2 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor2)->GetTensor();
        // input shape tensor
        auto in_shape_tensor = shapeOf(*input_tensors[0]);
        // hw shape tensor
        auto size = ShapeTensor(*input_tensor2);
        // get nc shape tensor
        DimsVector nc_axes = {0, 1};
        auto nc_index = ShapeTensor(1, std::move(nc_axes));
        auto nc = gather(network, in_shape_tensor, nc_index);
        // concat nc and hw
        out_shape_tensor = concat(network, nc, size);
    }


    // Dim Mode Special Case:
    // Cases When Both N,C and H+W are dynamic
    // In this case, We cannot turn to Scale mode.
    // Also layer->SetOutputDimensions() API does not accept -1 as dim
    // Have to use TNN Upsample Plugin.
    // e.g [-1,2,-1,-1]
    if (input_blobs_.size() == 1 && !paramlist->dims.empty()) {
        // In this case, network->addResize should not be called. GO Plugin
        auto trt_dim = input_tensor->getDimensions();
        if (trt_dim.d[0] <= 0 || trt_dim.d[1] <= 0) {
            LOGI("WARNING: Dynamic NCHW Upsample with fixed dims param is  NOT SUPPORTED by TensorRT, use TNN Upsample Plugin instead.\n");
            return TensorRTPluginLayerBuilder::AddToNetwork(network); 
        }
    }

    IResizeLayer* layer = network->addResize(*input_tensor);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
        if (input_blobs_.size() == 1) {
            if (!paramlist->dims.empty()) {
                auto trt_dim = input_tensor->getDimensions();
                if (trt_dim.nbDims != 4) {
                    LOGE("Upsample with 1 input only support 4d input.\n");
                    return nullptr;
                }

                // trt_dim may have one of the following values:
                // [-1,3,32,32], [-1,2,-1,-1], [1,16,256,256]
                if (trt_dim.d[0] <= 0 || trt_dim.d[1] <= 0) {
                    // Cases When At least One of N, C be dynamic
                    // and H,W are fixed, turn to scale mode
                    // Here trt_dim.d[2] > 0 && trt_dim.d[3] > 0
                    // e.g [-1,3,32,32]
                    float scale[4];
                    scale[0] = 1;
                    scale[1] = 1;
                    scale[2] = paramlist->dims[0] / float(trt_dim.d[2]);
                    scale[3] = paramlist->dims[1] / float(trt_dim.d[3]);
                    layer->setScales(scale, 4);
                } else {
                    // Cases When Both N and C are fixed
                    // e.g [1,16,256,256]
                    if (!output_dims.empty() && output_dims[2] > 0 && output_dims[3] > 0) {
                        nvinfer1::Dims4 dims(trt_dim.d[0], trt_dim.d[1], 
                            output_dims[2], output_dims[3]);
                        layer->setOutputDimensions(dims);
                    } else if (paramlist->dims.size() >= 2 && 
                               paramlist->dims[0] > 0 && paramlist->dims[1] > 0) {
                        nvinfer1::Dims4 dims(trt_dim.d[0], trt_dim.d[1], 
                            paramlist->dims[0], paramlist->dims[1]);
                        layer->setOutputDimensions(dims);
                    } else {
                        LOGE("Upsample with 1 input Fix N,C + Fixed dims does not have standard positive dim, Unsupported.\n");
                        return nullptr;
                    }
                }
            } else {
                if (output_dims.size() == 4) {
                    float scale[4];
                    scale[0] = 1;
                    scale[1] = 1;
                    scale[2] = paramlist->scales[1];
                    scale[3] = paramlist->scales[0];
                    layer->setScales(scale, 4);
                } else if (output_dims.size() == 5) {
                    float scale[5];
                    scale[0] = 1;
                    scale[1] = 1;
                    scale[2] = paramlist->scales[2];
                    scale[3] = paramlist->scales[1];
                    scale[4] = paramlist->scales[0];
                    layer->setScales(scale, 5);
                } else {
                    LOGE("Upsample with 1 input and scale param only support 2d or 3d now.\n");
                    return nullptr;
                }
            }
        } else if (input_blobs_.size() == 2) {
            // set resize layer input with shape tensor
            layer->setInput(1, out_shape_tensor.tensor(network));
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

