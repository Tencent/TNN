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

TensorRTLayerBuilder::TensorRTLayerBuilder(LayerType type) : TensorRTBaseLayerBuilder(type) {
    is_plugin = false;
}

TensorRTLayerBuilder::~TensorRTLayerBuilder() {
}

Status TensorRTLayerBuilder::Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& input_blobs,
        std::vector<Blob*>& output_blobs, AbstractDevice* device) {
    Status ret = m_layer->Init(context, param, resource, input_blobs, output_blobs, GetDevice(DEVICE_CUDA));
    if (ret != TNN_OK) {
        return ret;
    }

    input_blobs_  = m_layer->GetInputBlobs();
    output_blobs_ = m_layer->GetOutputBlobs();

    param_    = param;
    resource_ = resource;

    return TNN_OK;
}

Status TensorRTLayerBuilder::Forward() {
    return TNN_OK;
}

ILayer* TensorRTLayerBuilder::AddInt8OutputQDQLayers(ITensor* tensor, float quant_scale, float dequant_scale) {
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
    std::dynamic_pointer_cast<TensorRTTensor>(output_foreign_tensor)->SetQuantized();
    return output_dequant_layer;
}

ILayer* TensorRTLayerBuilder::AddInt8WeightQDQLayers(RawBuffer* weight, RawBuffer* bias, float scale, std::vector<int> dims) {
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
    weightDims.d[0] = dims[0];
    weightDims.d[1] = dims[1];
    weightDims.d[2] = dims[2];
    weightDims.d[3] = dims[3];
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

}  //  namespace TNN_NS
