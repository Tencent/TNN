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

#include "tnn/optimizer/net_optimizer_convert_int8_layers.h"

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"

namespace TNN_NS {

namespace optimizer {

    // Top priority: convert before all fuse
    NetOptimizerRegister<NetOptimizerConvertInt8Layers> g_net_optimizer_convert_int8_layers(OptPriority::P0);
    static const std::string int8_resource_name_suffix = "_scale_data_";

    std::string NetOptimizerConvertInt8Layers::Strategy() {
        return kNetOptimizerConvertInt8Layers;
    }

    bool NetOptimizerConvertInt8Layers::IsSupported(const NetworkConfig &net_config) {
        auto device = net_config.device_type;
        if (device == DEVICE_ARM || device == DEVICE_NAIVE) {
            one_direction_layers_.insert(LAYER_RELU);
            one_direction_layers_.insert(LAYER_UPSAMPLE);
            two_direction_layers_.insert(LAYER_ADD);
            two_direction_layers_.insert(LAYER_CONCAT);
            return true;
        }
        return false;
    }

    Status NetOptimizerConvertInt8Layers::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();
        if (count <= 1) {
            return TNN_OK;
        }

        // only used for quantized networks
        auto is_quantized_net = GetQuantizedInfoFromNetStructure(structure);
        if (!is_quantized_net) {
            return TNN_OK;
        }

        int layers_converted;
        // convert one-direction-layers until no convert is available
        while (1) {
            RETURN_ON_NEQ(ConvertByOneDirection(layers_orig, resource, layers_converted), TNN_OK);
            if (layers_converted > 0) {
                LOGD("%s converts %d one direction layers\n", kNetOptimizerConvertInt8Layers.c_str(), layers_converted);
            } else {
                break;
            }
        }

        // convert two-direction-layers once
        RETURN_ON_NEQ(ConvertByTwoDirection(layers_orig, resource, layers_converted), TNN_OK);
        if (layers_converted > 0) {
            LOGD("%s converts %d two direction layers\n", kNetOptimizerConvertInt8Layers.c_str(), layers_converted);
        }

        return TNN_OK;
    }

    Status NetOptimizerConvertInt8Layers::ConvertByOneDirection(
        const std::vector<std::shared_ptr<LayerInfo>> &layers_orig, NetResource *resource, int &count) {
        count = 0;
        for (int index = 0; index < layers_orig.size(); index++) {
            auto cur_layer = layers_orig[index];

            if (one_direction_layers_.count(cur_layer->type) == 0 || cur_layer->param->quantized) {
                continue;
            }

            bool can_convert_to_int8 = true;
            // assume inputs and outputs are one-one correspondence
            if (cur_layer->outputs.size() != cur_layer->inputs.size()) {
                continue;
            }
            auto &resource_map = resource->resource_map;
            for (int i = 0; i < cur_layer->outputs.size(); ++i) {
                if (resource_map.find(cur_layer->inputs[i] + int8_resource_name_suffix) == resource_map.end() &&
                    resource_map.find(cur_layer->outputs[i] + int8_resource_name_suffix) == resource_map.end()) {
                    can_convert_to_int8 = false;
                    break;
                }
            }
            if (can_convert_to_int8) {
                // set int resources
                for (int i = 0; i < cur_layer->outputs.size(); ++i) {
                    auto k_input_resource  = cur_layer->inputs[i] + int8_resource_name_suffix;
                    auto k_output_resource = cur_layer->outputs[i] + int8_resource_name_suffix;
                    if (resource_map.find(k_input_resource) == resource_map.end() &&
                        resource_map.find(k_output_resource) != resource_map.end()) {
                        resource_map[k_input_resource] = resource_map[k_output_resource];
                    } else if (resource_map.find(k_input_resource) != resource_map.end() &&
                               resource_map.find(k_output_resource) == resource_map.end()) {
                        resource_map[k_output_resource] = resource_map[k_input_resource];
                    } else if (resource_map.find(k_input_resource) == resource_map.end() &&
                               resource_map.find(k_output_resource) == resource_map.end()) {
                        return Status(TNNERR_LAYER_ERR, kNetOptimizerConvertInt8Layers + " internal error");
                    }
                }
                // convert to int8 layer
                std::string type_name = "Quantized" + cur_layer->type_str;
                cur_layer->type = GlobalConvertLayerType(type_name);
                cur_layer->type_str = type_name;
                cur_layer->param->quantized = true;
                LOGD("Convert to int8 layer: type %s name %s\n", cur_layer->type_str.c_str(), cur_layer->name.c_str());
                ++count;
            }
        }
        return TNN_OK;
    }

    Status NetOptimizerConvertInt8Layers::ConvertByTwoDirection(
        const std::vector<std::shared_ptr<LayerInfo>> &layers_orig, NetResource *resource, int &count) {
        count = 0;
        for (int index = 0; index < layers_orig.size(); index++) {
            auto cur_layer = layers_orig[index];

            if (two_direction_layers_.count(cur_layer->type) == 0 || cur_layer->param->quantized) {
                continue;
            }

            bool can_convert_to_int8 = true;
            // int resources of inputs and outputs should be available
            auto &resource_map = resource->resource_map;
            for (const auto &blob_name : cur_layer->inputs) {
                if (resource_map.find(blob_name + int8_resource_name_suffix) == resource_map.end()) {
                    can_convert_to_int8 = false;
                    break;
                }
            }
            for (const auto &blob_name : cur_layer->outputs) {
                if (resource_map.find(blob_name + int8_resource_name_suffix) == resource_map.end()) {
                    can_convert_to_int8 = false;
                    break;
                }
            }
            if (can_convert_to_int8) {
                // convert to int8 layer
                std::string type_name = "Quantized" + cur_layer->type_str;
                cur_layer->type = GlobalConvertLayerType(type_name);
                cur_layer->type_str = type_name;
                cur_layer->param->quantized = true;
                LOGD("Convert to int8 layer: type %s name %s\n", cur_layer->type_str.c_str(), cur_layer->name.c_str());
                ++count;
            }
        }
        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
