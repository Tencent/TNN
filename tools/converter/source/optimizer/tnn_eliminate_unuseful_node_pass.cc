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

#include <algorithm>

#include "tnn/interpreter/tnn/objseri.h"
#include "tnn_optimize_pass.h"

namespace TNN_CONVERTER {

DECLARE_OPTIMIZE_PASS(EliminateUnusefulNode);

std::string TnnOptimizeEliminateUnusefulNodePass::PassName() {
    return "EliminateUnusefulNode";
}

TNN_NS::Status TnnOptimizeEliminateUnusefulNodePass::exec(TNN_NS::NetStructure& net_structure,
                                                          TNN_NS::NetResource& net_resource) {
    std::set<std::string> black_list = {"Squeeze", "QuantizedPermute", "Int8Quantized", "Int8Dequantized"};
    auto& layers                     = net_structure.layers;
    for (auto iter = layers.begin(); iter != layers.end();) {
        auto& layer = *iter;
        if (black_list.find(layer->type_str) == black_list.end()) {
            iter++;
            continue;
        }
        // 处理悬空节点
        if (layer->inputs.empty() || layer->outputs.empty()) {
            iter = layers.erase(iter);
            continue;
        }

        auto input        = layer->inputs[0];
        auto output_names = layer->outputs;
        for (auto sub_iter = layers.begin(); sub_iter != layers.end(); sub_iter++) {
            auto& sub_layer = *sub_iter;
            for (int i = 0; i < sub_layer->inputs.size(); i++) {
                if (std::find(output_names.begin(), output_names.end(), sub_layer->inputs[i]) != output_names.end()) {
                    sub_layer->inputs[i] = input;
                }
            }
        }
        auto& model_outputs = net_structure.outputs;
        for (const auto& iter : model_outputs) {
            if (std::find(output_names.begin(), output_names.end(), iter) != output_names.end()) {
                model_outputs.erase(iter);
                model_outputs.insert(input);
                break;
            }
        }
        // erase blob scale
        auto& resource_map = net_resource.resource_map;
        if (layer->type_str == "QuantizedPermute" || layer->type_str == "Int8Quantized" ){
            for (const auto& output_name : output_names) {
                auto output_blob_scale_name = output_name + BLOB_SCALE_SUFFIX;
                if (resource_map.find(output_blob_scale_name) != resource_map.end()) {
                    resource_map.erase(output_blob_scale_name);
                }
            }
        }
//        else if (layer->type_str == "Int8Quantized") {
//            if (net_structure.inputs_shape_map.find(input) == net_structure.inputs_shape_map.end()) {
//               auto input_blob_scale_name = input + BLOB_SCALE_SUFFIX;
//               if (resource_map.find(input_blob_scale_name) != resource_map.end()) {
//                   resource_map.erase(input_blob_scale_name);
//               }
//            }
//        }

        iter = layers.erase(iter);
    }
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_OPTIMIZE_PASS(EliminateUnusefulNode);
}  // namespace TNN_CONVERTER
