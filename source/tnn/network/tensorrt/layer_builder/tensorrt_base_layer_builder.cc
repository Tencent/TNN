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

#include <mutex>

#include "tnn/network/tensorrt/layer_builder/tensorrt_base_layer_builder.h"
#include "tnn/network/tensorrt/utils.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

TensorRTBaseLayerBuilder::TensorRTBaseLayerBuilder(LayerType type) : BaseLayerBuilder(type) {
    m_layer = std::shared_ptr<BaseLayer>(CreateLayer(type));
}

TensorRTBaseLayerBuilder::~TensorRTBaseLayerBuilder() {
    for (int i = 0; i < int8_weight_data.size(); i++) {
        if (int8_weight_data[i]) {
            free(int8_weight_data[i]);
        }
    }
}

Status TensorRTBaseLayerBuilder::Reshape() {
    return TNN_OK;
}

Status TensorRTBaseLayerBuilder::InferOutputShape() {
    return TNN_OK;
}

Status TensorRTBaseLayerBuilder::Build() {
    return TNN_OK;
}

bool TensorRTBaseLayerBuilder::IsPluginLayer() {
    return this->is_plugin;
}

void TensorRTBaseLayerBuilder::SetConstantResource(ConstantResource* consts) {
    BaseLayer::SetConstantResource(consts);
    this->m_layer->SetConstantResource(consts);
}

void TensorRTBaseLayerBuilder::SetNetwork(TensorRTNetwork_* network) {
    this->m_network = network;
}

std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>& GetTRTLayerBuilderCreatorMap() {
    // static shared_ptr of LayerCreatorMap.
    static std::once_flag once;
    static std::shared_ptr<std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>); });
    return *creators;
}

std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>& GetTRTPluginLayerBuilderCreatorMap() {
    // static shared_ptr of LayerCreatorMap.
    static std::once_flag once;
    static std::shared_ptr<std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>); });
    return *creators;
}

TensorRTBaseLayerBuilder* CreateTensorRTBaseLayerBuilder(LayerType type) {
    TensorRTBaseLayerBuilder* cur_layer = nullptr;
    auto& trt_map = GetTRTLayerBuilderCreatorMap();
    auto& plugin_map = GetTRTPluginLayerBuilderCreatorMap();
    if (trt_map.count(type) > 0) {
        auto base_layer = trt_map[type]->CreateLayerBuilder();
        cur_layer = dynamic_cast<TensorRTBaseLayerBuilder*>(base_layer);
    } else if (plugin_map.count(type) > 0) {
        auto base_layer = plugin_map[type]->CreateLayerBuilder();
        cur_layer = dynamic_cast<TensorRTBaseLayerBuilder*>(base_layer);
    }
    return cur_layer;
}

ILayer* TensorRTBaseLayerBuilder::AddInt8OutputQDQLayers(nvinfer1::INetworkDefinition* network, ITensor* tensor,
        std::shared_ptr<ForeignTensor> foreign_tensor, float quant_scale, float dequant_scale) {
    Weights output_quant_shift;
    output_quant_shift.type = nvinfer1::DataType::kFLOAT;
    output_quant_shift.values = nullptr;
    output_quant_shift.count = 0;

    Weights output_quant_scale;
    output_quant_scale.type = nvinfer1::DataType::kFLOAT;
    float* output_quant_scale_data = (float*)malloc(sizeof(float));
    int8_weight_data.push_back(output_quant_scale_data);
    *output_quant_scale_data = quant_scale;
    output_quant_scale.values = (void*)output_quant_scale_data;
    output_quant_scale.count = 1;

    Weights output_quant_power;
    output_quant_power.type = nvinfer1::DataType::kFLOAT;
    output_quant_power.values = nullptr;
    output_quant_power.count = 0;

    ILayer* output_quant_layer = network->addScale(*tensor, ScaleMode::kUNIFORM,
        output_quant_shift, output_quant_scale, output_quant_power);
    std::string output_quant_layer_name = layer_name_ + "_output_quant_";
    output_quant_layer->setOutputType(0, nvinfer1::DataType::kINT8);
    output_quant_layer->setName(output_quant_layer_name.c_str());

    Weights output_dequant_shift;
    output_dequant_shift.type = nvinfer1::DataType::kFLOAT;
    output_dequant_shift.values = nullptr;
    output_dequant_shift.count = 0;

    Weights output_dequant_scale;
    output_dequant_scale.type = nvinfer1::DataType::kFLOAT;
    float* output_dequant_scale_data = (float*)malloc(sizeof(float));
    int8_weight_data.push_back(output_dequant_scale_data);
    *output_dequant_scale_data = dequant_scale;
    output_dequant_scale.values = (void*)output_dequant_scale_data;
    output_dequant_scale.count = 1;

    Weights output_dequant_power;
    output_dequant_power.type = nvinfer1::DataType::kFLOAT;
    output_dequant_power.values = nullptr;
    output_dequant_power.count = 0;

    ILayer* output_dequant_layer = network->addScale(*(output_quant_layer->getOutput(0)), ScaleMode::kUNIFORM,
        output_dequant_shift, output_dequant_scale, output_dequant_power);
    std::string output_dequant_layer_name = layer_name_ + "_output_dequant_";
    output_dequant_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
    output_dequant_layer->setName(output_dequant_layer_name.c_str());
    std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->SetQuantized();

    return output_dequant_layer;
}

ILayer* TensorRTBaseLayerBuilder::AddInt8WeightQDQLayers(nvinfer1::INetworkDefinition* network,
        RawBuffer* weight, nvinfer1::Weights &kernelWeights, RawBuffer* bias, nvinfer1::Weights &biasWeights,
        float scale, std::vector<int> dims) {
    kernelWeights.type = nvinfer1::DataType::kFLOAT;
    kernelWeights.values = nullptr;
    kernelWeights.count = 0;

    Weights int8Weights;
    int8Weights.type = nvinfer1::DataType::kFLOAT;
    float* host_weight = (float*)malloc(weight->GetDataCount() * sizeof(float));
    int8_weight_data.push_back(host_weight);
    for (int i = 0; i < weight->GetDataCount(); i++) {
        host_weight[i] = weight->force_to<int8_t*>()[i];
    }
    int8Weights.values = (void*)host_weight;
    int8Weights.count = weight->GetDataCount();

    biasWeights.type = nvinfer1::DataType::kFLOAT;
    if (bias) {
        float* host_bias = (float*)malloc(bias->GetDataCount() * sizeof(float));
        int8_weight_data.push_back(host_bias);
        for (int i = 0; i < bias->GetDataCount(); i++) {
            host_bias[i] = (bias->force_to<int*>())[i];
        }
        biasWeights.values = (void*)host_bias;
        biasWeights.count = bias->GetDataCount();
    } else {
        biasWeights.values = nullptr;
        biasWeights.count = 0;
    }

    Dims weightDims;
    weightDims.nbDims = dims.size();
    for (size_t i = 0; i < dims.size(); i++) {
        weightDims.d[i] = dims[i];
    }

    ILayer* constant_layer = network->addConstant(weightDims, int8Weights);

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

    ILayer* weight_quant_layer = network->addScale(*(constant_layer->getOutput(0)), ScaleMode::kUNIFORM,
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
    *weight_dequant_scale_data = scale;
    int8_weight_data.push_back(weight_dequant_scale_data);
    weight_dequant_scale.type = nvinfer1::DataType::kFLOAT;
    weight_dequant_scale.values = (void*)weight_dequant_scale_data;
    weight_dequant_scale.count = 1;

    Weights weight_dequant_power;
    weight_dequant_power.type = nvinfer1::DataType::kFLOAT;
    weight_dequant_power.values = nullptr;
    weight_dequant_power.count = 0;

    ILayer* weight_dequant_layer = network->addScale(*(weight_quant_layer->getOutput(0)), ScaleMode::kUNIFORM,
        weight_dequant_shift, weight_dequant_scale, weight_dequant_power);
    std::string weight_dequant_layer_name = layer_name_ + "_weight_dequant_";
    weight_dequant_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
    weight_dequant_layer->setName(weight_dequant_layer_name.c_str());
    return weight_dequant_layer;
}

std::vector<ITensor*> TensorRTBaseLayerBuilder::GetInputITensors() {
    std::vector<ITensor *> inputs;
    for(auto blob : input_blobs_) {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(blob)->GetForeignTensor();
        if (foreign_tensor) {
            auto tensorrt_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor);
            if (tensorrt_tensor) {
                if (nullptr == tensorrt_tensor->GetTensor()) {
                    LOGE("InputITensors[%d]:%s got nullptr for layer %s\n", 
                                        inputs.size(), blob->GetBlobDesc().name.c_str(), GetLayerName().c_str());
                }
                inputs.push_back(tensorrt_tensor->GetTensor());
            } else {
                LOGE("GetInputITensors got non-TensorRTTensor\n");
            }
        } else {
            LOGE("GetInputITensors got non-ForeignBlob\n");
        }
    }
    return inputs;
}

std::vector<ITensor*> TensorRTBaseLayerBuilder::GetOutputITensors() {
    std::vector<ITensor *> outputs;
    for(auto blob : output_blobs_) {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(blob)->GetForeignTensor();
        if (foreign_tensor) {
            auto tensorrt_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor);
            if (tensorrt_tensor) {
                outputs.push_back(tensorrt_tensor->GetTensor());
            } else {
                LOGE("GetOutputITensors got non-TensorRTTensor\n");
            }
        } else {
            LOGE("GetOutputITensors got non-ForeignBlob\n");
        }
    }
    return outputs;   
}

}  //  namespace TNN_NS
