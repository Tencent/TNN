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

#include "dynamic_range_quantization.h"

namespace TNN_NS {
DynamicRangeQuantizer::DynamicRangeQuantizer(const std::shared_ptr<NetStructure>& net_structure,
                                             const std::shared_ptr<NetResource>& net_resource) {
    net_structure_ = net_structure;
    net_resource_  = net_resource;
}

Status DynamicRangeQuantizer::GetDynamicRangeQuantModel(std::shared_ptr<NetStructure>& net_structure,
                                                        std::shared_ptr<NetResource>& net_resource) {
    net_structure  = std::make_shared<NetStructure>();
    *net_structure = *net_structure_;
    net_resource   = std::make_shared<NetResource>();
    *net_resource  = *net_resource_;

    std::set<LayerType> support_layer = {LAYER_CONVOLUTION, LAYER_LSTM, LAYER_MATMUL};

    auto& net_layers   = net_structure->layers;
    auto& resource_map = net_resource->resource_map;
    auto& constant_map = net_resource->constant_map;
    for (auto& layer : net_layers) {
        const auto type = layer->type;
        switch (type) {
            case LAYER_CONVOLUTION:
                QuantConvolution(layer, resource_map, constant_map);
                break;
            case LAYER_LSTMONNX:
                QuantLSTM(layer, resource_map, constant_map);
                break;
            case LAYER_MATMUL:
                QuantMatMul(layer, resource_map, constant_map);
                break;
            default:
                break;
        }
    }

    return TNN_OK;
}

Status DynamicRangeQuantizer::QuantConvolution(std::shared_ptr<LayerInfo>& layer,
                                               std::map<std::string, std::shared_ptr<LayerResource>>& resource_map,
                                               std::map<std::string, std::shared_ptr<RawBuffer>>& constant_map) {
    auto conv_param                               = std::dynamic_pointer_cast<ConvLayerParam>(layer->param);
    conv_param->dynamic_range_quantized           = true;
    std::shared_ptr<LayerResource> layer_resource = nullptr;
    if (resource_map.find(layer->name) != resource_map.end()) {
        layer_resource = resource_map[layer->name];
    }

    RawBuffer quant_buf;
    RawBuffer scale_buf;
    auto conv_resource = std::dynamic_pointer_cast<ConvLayerResource>(layer_resource);
    PerChannelQuant(conv_resource->filter_handle, quant_buf, scale_buf, conv_param->output_channel);

    conv_resource->filter_handle = quant_buf;
    conv_resource->scale_handle  = scale_buf;

    return TNN_OK;
}

Status DynamicRangeQuantizer::QuantLSTM(std::shared_ptr<LayerInfo>& layer,
                                        std::map<std::string, std::shared_ptr<LayerResource>>& resource_map,
                                        std::map<std::string, std::shared_ptr<RawBuffer>>& constant_map) {
    layer->param->dynamic_range_quantized = true;

    RawBuffer weight_buf;
    RawBuffer recurrence_buf;
    auto weight_name     = layer->inputs.at(1);
    auto recurrence_name = layer->inputs.at(2);
    if (constant_map.find(weight_name) != constant_map.end()) {
        weight_buf = *constant_map[weight_name];
    }
    if (constant_map.find(recurrence_name) != constant_map.end()) {
        recurrence_buf = *constant_map[recurrence_name];
    }

    std::shared_ptr<RawBuffer> quant_weight_buf     = std::make_shared<RawBuffer>();
    std::shared_ptr<RawBuffer> scale_weight_buf     = std::make_shared<RawBuffer>();
    std::shared_ptr<RawBuffer> quant_recurrence_buf = std::make_shared<RawBuffer>();
    std::shared_ptr<RawBuffer> scale_recurrence_buf = std::make_shared<RawBuffer>();
    PerTensorQuant(weight_buf, *quant_weight_buf, *scale_weight_buf);
    PerTensorQuant(recurrence_buf, *quant_recurrence_buf, *scale_recurrence_buf);

    constant_map[weight_name]                                    = quant_weight_buf;
    constant_map[recurrence_name]                                = quant_recurrence_buf;
    constant_map[weight_name + DynamicRangeQuantScaleSuffix]     = scale_weight_buf;
    constant_map[recurrence_name + DynamicRangeQuantScaleSuffix] = scale_recurrence_buf;

    return TNN_OK;
}

Status DynamicRangeQuantizer::QuantMatMul(std::shared_ptr<LayerInfo>& layer,
                                          std::map<std::string, std::shared_ptr<LayerResource>>& resource_map,
                                          std::map<std::string, std::shared_ptr<RawBuffer>>& constant_map) {
    auto matmul_param                     = std::dynamic_pointer_cast<MatMulLayerParam>(layer->param);
    matmul_param->dynamic_range_quantized = true;
    if (matmul_param->weight_position != 1) {
        return TNN_OK;
    }
    std::shared_ptr<LayerResource> layer_resource = nullptr;
    if (resource_map.find(layer->name) != resource_map.end()) {
        layer_resource = resource_map[layer->name];
    }

    RawBuffer quant_buf;
    RawBuffer scale_buf;
    auto matmul_resource = std::dynamic_pointer_cast<MatMulLayerResource>(layer_resource);
    PerTensorQuant(matmul_resource->weight, quant_buf, scale_buf);

    matmul_resource->weight       = quant_buf;
    matmul_resource->scale_handle = scale_buf;

    return TNN_OK;
}

Status DynamicRangeQuantizer::PerChannelQuant(RawBuffer& weight_buf, RawBuffer& quant_buf, RawBuffer& scale_buf,
                                              int num_kernel) {
    const int weight_size = weight_buf.GetDataCount();
    const int kernel_size = weight_size / num_kernel;
    auto weight_data_ptr  = weight_buf.force_to<float*>();

    std::vector<int8_t> quant_data(weight_size, 0);
    std::vector<float> scale_data(num_kernel, 0.0f);

    int begin_index = 0;
    for (int k = 0; k < num_kernel; k++) {
        begin_index    = k * kernel_size;
        auto max_value = GetAbsMax(weight_data_ptr + begin_index, kernel_size);
        scale_data[k]  = max_value / threshold_;
        for (int i = 0; i < kernel_size; i++) {
            quant_data[begin_index + i] = int8_t(std::round(weight_data_ptr[begin_index + i] / scale_data[k]));
        }
    }

    quant_buf = RawBuffer(weight_size * sizeof(int8_t));
    memcpy(quant_buf.force_to<int8_t*>(), quant_data.data(), weight_size * sizeof(int8_t));
    quant_buf.SetDataType(DATA_TYPE_INT8);
    quant_buf.SetBufferDims(weight_buf.GetBufferDims());

    scale_buf = RawBuffer(num_kernel * sizeof(float));
    memcpy(scale_buf.force_to<float*>(), scale_data.data(), num_kernel * sizeof(float));
    scale_buf.SetDataType(DATA_TYPE_FLOAT);
    scale_buf.SetBufferDims({num_kernel});

    return TNN_OK;
}
Status DynamicRangeQuantizer::PerTensorQuant(RawBuffer& weight_buf, RawBuffer& quant_buf, RawBuffer& scale_buf) {
    const int weight_size = weight_buf.GetDataCount();
    auto weight_data_ptr  = weight_buf.force_to<float*>();
    auto max_value        = GetAbsMax(weight_data_ptr, weight_size);
    auto scale_data       = max_value / threshold_;

    std::vector<int8_t> quant_data(weight_size, 0);
    for (int i = 0; i < weight_size; i++) {
        quant_data[i] = int8_t(std::round(weight_data_ptr[i] / scale_data));
    }

    int x     = sizeof(int8_t);
    quant_buf = RawBuffer(weight_size * x);
    memcpy(quant_buf.force_to<int8_t*>(), quant_data.data(), weight_size * sizeof(int8_t));
    quant_buf.SetDataType(DATA_TYPE_INT8);
    quant_buf.SetBufferDims(weight_buf.GetBufferDims());

    scale_buf = RawBuffer(sizeof(float));
    memcpy(scale_buf.force_to<float*>(), &scale_data, sizeof(float));
    scale_buf.SetDataType(DATA_TYPE_FLOAT);
    scale_buf.SetBufferDims({1});

    return TNN_OK;
}

float DynamicRangeQuantizer::GetAbsMax(float* data, int data_size) {
    float max_value = fabs(data[0]);
    for (int i = 1; i < data_size; i++) {
        float value = fabs(data[i]);
        if (value > max_value) {
            max_value = value;
        }
    }

    return max_value;
}
}  // namespace TNN_NS
