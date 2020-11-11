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
    ILayer* last_layer;
    if (int8) {
        float weight_scale_value = *(resource->scale_handle.force_to<float*>());
        float input_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        kernelWeights.type = nvinfer1::DataType::kFLOAT;
        kernelWeights.values = nullptr;
        kernelWeights.count = 0;

        Weights int8Weights;
        int8Weights.type = nvinfer1::DataType::kFLOAT;
        float* host_weight = (float*)malloc(resource->weight_handle.GetDataCount() * sizeof(float));
        int8_weight_data.push_back(host_weight);
        for (int i = 0; i < resource->weight_handle.GetDataCount(); i++) {
            host_weight[i] = resource->weight_handle.force_to<int8_t*>()[i];
        }
        int8Weights.values = (void*)host_weight;
        int8Weights.count = resource->weight_handle.GetDataCount();

        biasWeights.type = nvinfer1::DataType::kFLOAT;
        if (paramlist->has_bias) {
            float* host_bias = (float*)malloc(resource->bias_handle.GetDataCount() * sizeof(float));
            int8_weight_data.push_back(host_bias);
            for (int i = 0; i < resource->bias_handle.GetDataCount(); i++) {
                host_bias[i] = (resource->bias_handle.force_to<int*>())[i];
            }
            biasWeights.values = (void*)host_bias;
            biasWeights.count = resource->bias_handle.GetDataCount();
        } else {
            biasWeights.values = nullptr;
            biasWeights.count = 0;
        }

        Dims weightDims;
        weightDims.nbDims = 4;
        weightDims.d[0] = output_blobs_[0]->GetBlobDesc().dims[1];
        weightDims.d[1] = input_blobs_[0]->GetBlobDesc().dims[1];
        weightDims.d[2] = 1;
        weightDims.d[3] = 1;
        auto constant_layer = network->addConstant(weightDims, int8Weights);

        Weights weight_quant_shift;
        weight_quant_shift.type = nvinfer1::DataType::kFLOAT;
        weight_quant_shift.values = nullptr;
        weight_quant_shift.count = 0;

        Weights weight_quant_scale;
        float* weight_quant_scale_data = (float*)malloc(sizeof(float));
        int8_weight_data.push_back(weight_quant_scale_data);
        *weight_quant_scale_data = 1.f;
        weight_quant_scale.type = nvinfer1::DataType::kFLOAT;
        weight_quant_scale.values = (void*)weight_quant_scale_data;
        weight_quant_scale.count = 1;

        Weights weight_quant_power;
        weight_quant_power.type = nvinfer1::DataType::kFLOAT;
        weight_quant_power.values = nullptr;
        weight_quant_power.count = 0;

        auto weight_quant_layer = network->addScale(*(constant_layer->getOutput(0)), ScaleMode::kUNIFORM,
            weight_quant_shift, weight_quant_scale, weight_quant_power);
        std::string weight_quant_layer_name = layer_name_ + "_weight_quant_";
        weight_quant_layer->setOutputType(0, nvinfer1::DataType::kINT8);
        weight_quant_layer->setName(weight_quant_layer_name.c_str());

        Weights weight_dequant_shift;
        weight_dequant_shift.type = nvinfer1::DataType::kFLOAT;
        weight_dequant_shift.values = nullptr;
        weight_dequant_shift.count = 0;

        Weights weight_dequant_scale;
        float* weight_dequant_scale_data = (float*)malloc(sizeof(float));
        *weight_dequant_scale_data = 1 / (weight_scale_value / input_scale_value);
        int8_weight_data.push_back(weight_dequant_scale_data);
        weight_dequant_scale.type = nvinfer1::DataType::kFLOAT;
        weight_dequant_scale.values = (void*)weight_dequant_scale_data;
        weight_dequant_scale.count = 1;

        Weights weight_dequant_power;
        weight_dequant_power.type = nvinfer1::DataType::kFLOAT;
        weight_dequant_power.values = nullptr;
        weight_dequant_power.count = 0;

        auto weight_dequant_layer = network->addScale(*(weight_quant_layer->getOutput(0)), ScaleMode::kUNIFORM,
            weight_dequant_shift, weight_dequant_scale, weight_dequant_power);
        std::string weight_dequant_layer_name = layer_name_ + "_weight_dequant_";
        weight_dequant_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        weight_dequant_layer->setName(weight_dequant_layer_name.c_str());
        last_layer = weight_dequant_layer;
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

    IFullyConnectedLayer* fc_layer = network->addFullyConnected(*input_tensor, paramlist->num_output, kernelWeights, biasWeights);
    if (int8) {
        fc_layer->setInput(1, *(last_layer->getOutput(0)));
        fc_layer->setPrecision(nvinfer1::DataType::kINT8);
    }

    if (fc_layer != nullptr) {
        fc_layer->setName(layer_name_.c_str());
        last_layer = fc_layer;
    }

    if (int8) {
        float output_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(output_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        Weights output_quant_shift;
        output_quant_shift.type = nvinfer1::DataType::kFLOAT;
        output_quant_shift.values = nullptr;
        output_quant_shift.count = 0;

        Weights output_quant_scale;
        output_quant_scale.type = nvinfer1::DataType::kFLOAT;
        float* output_quant_scale_data = (float*)malloc(sizeof(float));
        int8_weight_data.push_back(output_quant_scale_data);
        *output_quant_scale_data = output_scale_value;
        output_quant_scale.values = (void*)output_quant_scale_data;
        output_quant_scale.count = 1;

        Weights output_quant_power;
        output_quant_power.type = nvinfer1::DataType::kFLOAT;
        output_quant_power.values = nullptr;
        output_quant_power.count = 0;

        auto output_quant_layer = network->addScale(*(last_layer->getOutput(0)), ScaleMode::kUNIFORM,
            output_quant_shift, output_quant_scale, output_quant_power);
        std::string output_scale_name1 = layer_name_ + "_output_quant_";
        output_quant_layer->setOutputType(0, nvinfer1::DataType::kINT8);
        output_quant_layer->setName(output_scale_name1.c_str());

        Weights output_dequant_shift;
        output_dequant_shift.type = nvinfer1::DataType::kFLOAT;
        output_dequant_shift.values = nullptr;
        output_dequant_shift.count = 0;

        Weights output_dequant_scale;
        output_dequant_scale.type = nvinfer1::DataType::kFLOAT;
        float* output_dequant_scale_data = (float*)malloc(sizeof(float));
        int8_weight_data.push_back(output_dequant_scale_data);
        *output_dequant_scale_data = 1 / output_scale_value;
        output_dequant_scale.values = (void*)output_dequant_scale_data;
        output_dequant_scale.count = 1;

        Weights output_dequant_power;
        output_dequant_power.type = nvinfer1::DataType::kFLOAT;
        output_dequant_power.values = nullptr;
        output_dequant_power.count = 0;

        auto output_dequant_layer = network->addScale(*(output_quant_layer->getOutput(0)), ScaleMode::kUNIFORM,
            output_dequant_shift, output_dequant_scale, output_dequant_power);
        std::string output_dequant_layer_name = layer_name_ + "_output_dequant_";
        output_dequant_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        output_dequant_layer->setName(output_dequant_layer_name.c_str());
        last_layer = output_dequant_layer;
    }

    return last_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(InnerProduct, LAYER_INNER_PRODUCT);

}  //  namespace TNN_NS
