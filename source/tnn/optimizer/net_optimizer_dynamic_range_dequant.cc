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

#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/optimizer_const.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/net_optimizer_dynamic_range_dequant.h"

namespace TNN_NS {

namespace optimizer {

    NetOptimizerRegister<NetOptimizerDynamicRangeDequant> g_net_optimizer_dynamic_range_dequant(OptPriority::P1);

    std::string NetOptimizerDynamicRangeDequant::Strategy() {
        return kNetOptimizerDynamicRangeDequant;
    }

    bool NetOptimizerDynamicRangeDequant::IsSupported(const NetworkConfig &net_config) {
        return true;
    }

    Status NetOptimizerDynamicRangeDequant::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        if (structure->layers.size() <= 1) {
            return TNN_OK;
        }

        for (auto &layer : structure->layers) {
            if (!layer->param->dynamic_range_quantized) {
                continue;
            }
            auto type = layer->type;
            switch (type) {
                case LAYER_CONVOLUTION:
                    DequantConv(layer, structure, resource);
                    break;
                case LAYER_LSTMONNX:
                    DequantLSTM(layer, structure, resource);
                    break;
                case LAYER_MATMUL:
                    DequantMatMul(layer, structure, resource);
                    break;
                default:
                    break;
            }
        }

        return TNN_OK;
    }

    Status NetOptimizerDynamicRangeDequant::DequantConv(std::shared_ptr<LayerInfo> &layer, NetStructure *structure,
                                                        NetResource *resource) {
        auto layer_name    = layer->name;
        auto conv_resource = std::dynamic_pointer_cast<ConvLayerResource>(resource->resource_map[layer_name]);
        auto filter_handle = conv_resource->filter_handle;
        auto scale_handler = conv_resource->scale_handle;

        const auto filter_dims = filter_handle.GetBufferDims();
        const int num_kernel   = filter_dims.at(0);
        const int filter_size  = filter_handle.GetDataCount();
        const int kernel_size  = filter_size / num_kernel;
        auto filter_ptr        = filter_handle.force_to<int8_t *>();
        auto scale_ptr         = scale_handler.force_to<float *>();
        std::vector<float> weight_data(filter_size, 0);
        for (int k = 0; k < num_kernel; k++) {
            int begin_index = k * kernel_size;
            for (int i = 0; i < kernel_size; i++) {
                weight_data[begin_index + i] = scale_ptr[k] * (float)(filter_ptr[begin_index + i]);
            }
        }
        RawBuffer weight_buf(filter_size * sizeof(float));
        memcpy(weight_buf.force_to<float *>(), weight_data.data(), filter_size * sizeof(float));
        weight_buf.SetDataType(DATA_TYPE_FLOAT);
        weight_buf.SetBufferDims(filter_dims);

        conv_resource->filter_handle = weight_buf;
        conv_resource->scale_handle  = RawBuffer();

        return TNN_OK;
    }

    Status NetOptimizerDynamicRangeDequant::DequantLSTM(std::shared_ptr<LayerInfo> &layer, NetStructure *structure,
                                                        NetResource *resource) {
        // dequant weight and recurrence
        for (int idx = 1; idx <= 2; idx++) {
            auto buffer_name = layer->inputs.at(idx);
            auto scale_name  = buffer_name + DynamicRangeQuantScaleSuffix;
            auto buffer      = resource->constant_map[buffer_name];
            auto scale       = resource->constant_map[scale_name];

            const int data_size = buffer->GetDataCount();
            auto weight_ptr     = buffer->force_to<int8_t *>();
            auto scale_value    = scale->force_to<float *>()[0];
            std::vector<float> weight_data(data_size, 0);
            for (int i = 0; i < data_size; i++) {
                weight_data[i] = scale_value * (float)(weight_ptr[i]);
            }

            auto weight_buf = std::make_shared<RawBuffer>(data_size * sizeof(float));
            memcpy(weight_buf->force_to<float *>(), weight_data.data(), data_size * sizeof(float));
            weight_buf->SetDataType(DATA_TYPE_FLOAT);
            weight_buf->SetBufferDims(buffer->GetBufferDims());

            resource->constant_map[buffer_name] = weight_buf;

            // delete scale buffer
            if (resource->constant_map.count(scale_name)) {
                resource->constant_map.erase(scale_name);
            }
        }
        return TNN_OK;
    }

    Status NetOptimizerDynamicRangeDequant::DequantMatMul(std::shared_ptr<LayerInfo> &layer, NetStructure *structure,
                                                          NetResource *resource) {
        auto layer_name      = layer->name;
        auto matmul_resource = std::dynamic_pointer_cast<MatMulLayerResource>(resource->resource_map[layer_name]);
        auto scale_handle    = matmul_resource->scale_handle;

        const int data_size = matmul_resource->weight.GetDataCount();
        auto weight_ptr     = matmul_resource->weight.force_to<int8_t *>();
        auto scale_value    = scale_handle.force_to<float *>()[0];
        std::vector<float> weight_data(data_size, 0);
        for (int i = 0; i < data_size; i++) {
            weight_data[i] = scale_value * (float)(weight_ptr[i]);
        }

        RawBuffer weight_buf(data_size * sizeof(float));
        memcpy(weight_buf.force_to<float *>(), weight_data.data(), data_size * sizeof(float));
        weight_buf.SetDataType(DATA_TYPE_FLOAT);
        weight_buf.SetBufferDims(matmul_resource->weight.GetBufferDims());

        matmul_resource->weight = weight_buf;

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
