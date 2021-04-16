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

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/tnn/objseri.h"
#include "tnn_optimize_pass.h"

namespace TNN_CONVERTER {

DECLARE_OPTIMIZE_PASS(TransformDequantized);

std::set<TNN_NS::LayerType> one_direction_layers_ = {TNN_NS::LAYER_RELU, TNN_NS::LAYER_UPSAMPLE};
std::set<TNN_NS::LayerType> two_direction_layers_ = {TNN_NS::LAYER_ADD, TNN_NS::LAYER_CONCAT};

std::string TnnOptimizeTransformDequantizedPass::PassName() {
    return "TransformDequantized";
}

TNN_NS::Status ConvertByOneDirection(const std::vector<std::shared_ptr<TNN_NS::LayerInfo>> &layers_orig,
                                     TNN_NS::NetResource &resource) {
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
        auto &resource_map = resource.resource_map;
        for (int i = 0; i < cur_layer->outputs.size(); ++i) {
            if (resource_map.find(cur_layer->inputs[i] + BLOB_SCALE_SUFFIX) == resource_map.end() &&
                resource_map.find(cur_layer->outputs[i] + BLOB_SCALE_SUFFIX) == resource_map.end()) {
                can_convert_to_int8 = false;
                break;
            }
        }
        if (can_convert_to_int8) {
            // set int resources
            for (int i = 0; i < cur_layer->outputs.size(); ++i) {
                auto k_input_resource  = cur_layer->inputs[i] + BLOB_SCALE_SUFFIX;
                auto k_output_resource = cur_layer->outputs[i] + BLOB_SCALE_SUFFIX;
                if (resource_map.find(k_input_resource) == resource_map.end() &&
                    resource_map.find(k_output_resource) != resource_map.end()) {
                    resource_map[k_input_resource] = resource_map[k_output_resource];
                } else if (resource_map.find(k_input_resource) != resource_map.end() &&
                           resource_map.find(k_output_resource) == resource_map.end()) {
                    resource_map[k_output_resource] = resource_map[k_input_resource];
                } else if (resource_map.find(k_input_resource) == resource_map.end() &&
                           resource_map.find(k_output_resource) == resource_map.end()) {
                    return TNN_NS::Status(TNN_NS::TNNERR_LAYER_ERR, "Converter: ConvertByOneDirection failed\n");
                }
            }
            // convert to int8 layer
            std::string type_name       = "Quantized" + cur_layer->type_str;
            cur_layer->type             = TNN_NS::GlobalConvertLayerType(type_name);
            cur_layer->type_str         = type_name;
            cur_layer->param->quantized = true;
            LOGD("Convert to int8 layer: type %s name %s\n", cur_layer->type_str.c_str(), cur_layer->name.c_str());
        }
    }
    return TNN_NS::TNN_CONVERT_OK;
}

TNN_NS::Status ConvertByTwoDirection(const std::vector<std::shared_ptr<TNN_NS::LayerInfo>> &layers_orig,
                                     TNN_NS::NetResource &resource) {
    for (int index = 0; index < layers_orig.size(); index++) {
        auto cur_layer = layers_orig[index];

        if (two_direction_layers_.count(cur_layer->type) == 0 || cur_layer->param->quantized) {
            continue;
        }
        bool can_convert_to_int8 = true;
        // int resources of inputs and outputs should be available
        auto &resource_map = resource.resource_map;
        for (const auto &blob_name : cur_layer->inputs) {
            if (resource_map.find(blob_name + BLOB_SCALE_SUFFIX) == resource_map.end()) {
                can_convert_to_int8 = false;
                break;
            }
        }
        for (const auto &blob_name : cur_layer->outputs) {
            if (resource_map.find(blob_name + BLOB_SCALE_SUFFIX) == resource_map.end()) {
                can_convert_to_int8 = false;
                break;
            }
        }
        if (can_convert_to_int8) {
            // convert to int8 layer
            std::string type_name       = "Quantized" + cur_layer->type_str;
            cur_layer->type             = TNN_NS::GlobalConvertLayerType(type_name);
            cur_layer->type_str         = type_name;
            cur_layer->param->quantized = true;
            LOGD("Convert to int8 layer: type %s name %s\n", cur_layer->type_str.c_str(), cur_layer->name.c_str());
        }
    }
    return TNN_NS::TNN_CONVERT_OK;
}

TNN_NS::Status TnnOptimizeTransformDequantizedPass::exec(TNN_NS::NetStructure &net_structure,
                                                         TNN_NS::NetResource &net_resource) {
    auto &layers          = net_structure.layers;
    auto is_quantized_net = TNN_NS::GetQuantizedInfoFromNetStructure(&net_structure);
    if (!is_quantized_net) {
        return TNN_NS::TNN_CONVERT_OK;
    }
    auto status = ConvertByOneDirection(layers, net_resource);
    ASSERT(status == TNN_NS::TNN_CONVERT_OK);
    status = ConvertByTwoDirection(layers, net_resource);
    ASSERT(status == TNN_NS::TNN_CONVERT_OK);
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_OPTIMIZE_PASS(TransformDequantized);
}  // namespace TNN_CONVERTER
