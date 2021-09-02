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

#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"
#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);

ILayer* InnerProductTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<InnerProductLayerParam*>(param_);
    auto resource = dynamic_cast<InnerProductLayerResource*>(resource_);

    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();
    bool int8 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetInt8Mode();

    Weights kernelWeights;
    Weights biasWeights;
    ILayer* weight_layer;
    if (int8) {
        float weight_scale_value = *(resource->scale_handle.force_to<float*>());
        float input_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(
            input_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        float output_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(
            output_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        std::vector<int> dims;
        dims.push_back(output_blobs_[0]->GetBlobDesc().dims[1]);
        dims.push_back(input_blobs_[0]->GetBlobDesc().dims[1]);
        dims.push_back(1);
        dims.push_back(1);
        weight_layer = AddInt8WeightQDQLayers(network, &(resource->weight_handle), kernelWeights,
            paramlist->has_bias ? &(resource->bias_handle) : nullptr,
            biasWeights, output_scale_value / (weight_scale_value / input_scale_value), dims);

        if (!std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->IsQuantized()) {
            Weights input_quant_shift;
            input_quant_shift.type = nvinfer1::DataType::kFLOAT;
            input_quant_shift.values = nullptr;
            input_quant_shift.count = 0;

            Weights input_quant_scale;
            input_quant_scale.type = nvinfer1::DataType::kFLOAT;
            float* input_quant_scale_data = (float*)malloc(sizeof(float));
            int8_weight_data.push_back(input_quant_scale_data);
            *input_quant_scale_data = input_scale_value;
            input_quant_scale.values = (void*)input_quant_scale_data;
            input_quant_scale.count = 1;

            Weights input_quant_power;
            input_quant_power.type = nvinfer1::DataType::kFLOAT;
            input_quant_power.values = nullptr;
            input_quant_power.count = 0;

            auto input_quant_layer = network->addScale(*input_tensor, ScaleMode::kUNIFORM,
                input_quant_shift, input_quant_scale, input_quant_power);
            std::string input_scale_name = layer_name_ + "_input_quant_";
            input_quant_layer->setOutputType(0, nvinfer1::DataType::kINT8);
            input_quant_layer->setName(input_scale_name.c_str());

            Weights input_dequant_shift;
            input_dequant_shift.type = nvinfer1::DataType::kFLOAT;
            input_dequant_shift.values = nullptr;
            input_dequant_shift.count = 0;

            Weights input_dequant_scale;
            input_dequant_scale.type = nvinfer1::DataType::kFLOAT;
            float* input_dequant_scale_data = (float*)malloc(sizeof(float));
            int8_weight_data.push_back(input_dequant_scale_data);
            *input_dequant_scale_data = 1 / input_scale_value;
            input_dequant_scale.values = (void*)input_dequant_scale_data;
            input_dequant_scale.count = 1;

            Weights input_dequant_power;
            input_dequant_power.type = nvinfer1::DataType::kFLOAT;
            input_dequant_power.values = nullptr;
            input_dequant_power.count = 0;

            auto input_dequant_layer = network->addScale(*(input_quant_layer->getOutput(0)), ScaleMode::kUNIFORM,
                input_dequant_shift, input_dequant_scale, input_dequant_power);
            std::string input_dequant_layer_name = layer_name_ + "_input_dequant_";
            input_dequant_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
            input_dequant_layer->setName(input_dequant_layer_name.c_str());
            std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->SetQuantized();
            input_tensor = input_dequant_layer->getOutput(0);
        }
    } else {
        kernelWeights.type = nvinfer1::DataType::kFLOAT;
        kernelWeights.values = resource->weight_handle.force_to<void*>();
        kernelWeights.count = resource->weight_handle.GetDataCount();
        if (paramlist->has_bias) {
            biasWeights.type = nvinfer1::DataType::kFLOAT;
            biasWeights.values = resource->bias_handle.force_to<void*>();
            biasWeights.count = resource->bias_handle.GetDataCount();
        } else {
            biasWeights.type = nvinfer1::DataType::kFLOAT;
            biasWeights.values = nullptr;
            biasWeights.count = 0;
        }
    }

    ILayer* layer;

    Dims in_dims;
    in_dims.nbDims = 4;
    in_dims.d[0] = -1;
    in_dims.d[1] = kernelWeights.count / paramlist->num_output;
    in_dims.d[2] = 1;
    in_dims.d[3] = 1;
    IShuffleLayer* in_reshape_layer = network->addShuffle(*input_tensor);
    in_reshape_layer->setReshapeDimensions(in_dims);
    input_tensor = in_reshape_layer->getOutput(0);

    //FullyConnected
    layer = network->addFullyConnected(*input_tensor, paramlist->num_output, 
        kernelWeights, biasWeights);
    if (int8) {
        layer->setInput(1, *(weight_layer->getOutput(0)));
        layer->setPrecision(nvinfer1::DataType::kINT8);
    }

    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
        input_tensor = layer->getOutput(0);
    }

    if (int8) {
        float output_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(
            output_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        auto output_dequant_layer =  AddInt8OutputQDQLayers(network, layer->getOutput(0), output_foreign_tensor, 1, 1 / output_scale_value);
        input_tensor = output_dequant_layer->getOutput(0);
    }

    Dims out_dims;
    out_dims.nbDims = paramlist->axis + 1;
    for (int i = 0; i < out_dims.nbDims; i++) {
        out_dims.d[i] = 0;
    }
    IShuffleLayer* out_reshape_layer = network->addShuffle(*input_tensor);
    out_reshape_layer->setReshapeDimensions(out_dims);
    layer = out_reshape_layer;

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);

}  //  namespace TNN_NS

