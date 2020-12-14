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

#include <vector>
#include <algorithm>

#include "tnn/network/tensorrt/layer_builder/activation_layer_builder.h"

namespace TNN_NS {

ActivationTRTLayerBuilder::ActivationTRTLayerBuilder(LayerType ignore) : TensorRTLayerBuilder(ignore) {
}

ILayer* ActivationTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();
    bool int8 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetInt8Mode();

    ILayer* last_layer;
    IActivationLayer* activation_layer = network->addActivation(*input_tensor, m_type);
    if (activation_layer != nullptr) {
        activation_layer->setName(layer_name_.c_str());
        if (m_type == nvinfer1::ActivationType::kCLIP) {
            auto paramlist = dynamic_cast<ClipLayerParam *>(param_);
            activation_layer->setAlpha(paramlist->min);
            activation_layer->setBeta(paramlist->max);
        }
        if (m_type == nvinfer1::ActivationType::kLEAKY_RELU) {
            auto resource = dynamic_cast<PReluLayerResource*>(resource_);
            auto scope = resource->slope_handle.force_to<float*>();
            layer->setAlpha(*scope);
        }
        last_layer = activation_layer;
    }

    if (int8) {
        float output_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(output_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        activation_layer->setPrecision(nvinfer1::DataType::kINT8);

        Weights output_quant_shift;
        output_quant_shift.type = nvinfer1::DataType::kFLOAT;
        output_quant_shift.values = nullptr;
        output_quant_shift.count = 0;

        Weights output_quant_scale;
        float* output_quant_scale_data = (float*)malloc(sizeof(float));
        int8_weight_data.push_back(output_quant_scale_data);
        *output_quant_scale_data = output_scale_value;
        output_quant_scale.type = nvinfer1::DataType::kFLOAT;
        output_quant_scale.values = (void*)output_quant_scale_data;
        output_quant_scale.count = 1;

        Weights output_quant_power;
        output_quant_power.type = nvinfer1::DataType::kFLOAT;
        output_quant_power.values = nullptr;
        output_quant_power.count = 0;

        auto output_quant_layer = network->addScale(*(last_layer->getOutput(0)), ScaleMode::kUNIFORM,
            output_quant_shift, output_quant_scale, output_quant_power);
        std::string output_quant_name = layer_name_ + "_output_quant_";
        output_quant_layer->setOutputType(0, nvinfer1::DataType::kINT8);
        output_quant_layer->setName(output_quant_name.c_str());

        Weights output_dequant_shift;
        output_dequant_shift.type = nvinfer1::DataType::kFLOAT;
        output_dequant_shift.values = nullptr;
        output_dequant_shift.count = 0;

        Weights output_dequant_scale;
        float* output_dequant_scale_data = (float*)malloc(sizeof(float));
        int8_weight_data.push_back(output_dequant_scale_data);
        *output_dequant_scale_data = 1 / output_scale_value;
        output_dequant_scale.type = nvinfer1::DataType::kFLOAT;
        output_dequant_scale.values = (void*)output_dequant_scale_data;
        output_dequant_scale.count = 1;

        Weights output_dequant_power;
        output_dequant_power.type = nvinfer1::DataType::kFLOAT;
        output_dequant_power.values = nullptr;
        output_dequant_power.count = 0;

        auto output_dequant_layer = network->addScale(*(output_quant_layer->getOutput(0)),
            ScaleMode::kUNIFORM, output_dequant_shift, output_dequant_scale, output_dequant_power);
        std::string output_dequant_name = layer_name_ + "_output_dequant_";
        output_dequant_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        output_dequant_layer->setName(output_dequant_name.c_str());
        last_layer = output_dequant_layer;
    }

    return last_layer;
}

}  //  namespace TNN_NS
