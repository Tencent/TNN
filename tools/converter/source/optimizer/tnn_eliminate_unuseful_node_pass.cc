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
    std::set<std::string> black_list = {"Int8Quantized", "Int8Dequantized"};
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
            iter++;
            continue;
        }
        if (layer->inputs.size() > 1) {
            iter++;
            continue;
        }
        auto pre_node_name      = layer->inputs[0];
        auto layer_output_names = layer->outputs;
        // to deal with the int8 dequantize node is the model output
        auto& model_output_names = net_structure.outputs;
        for (const auto& output_name : model_output_names) {
            if (std::find(layer_output_names.begin(), layer_output_names.end(), output_name) !=
                layer_output_names.end()) {
                model_output_names.erase(output_name);
                model_output_names.insert(pre_node_name);
                break;
            }
        }

        for (const auto& sub_iter : layers) {
            for (int i = 0; i < sub_iter->inputs.size(); i++) {
                if (std::find(layer_output_names.begin(), layer_output_names.end(), sub_iter->inputs[i]) !=
                    layer_output_names.end()) {
                    sub_iter->inputs[i] = pre_node_name;
                }
            }
        }
        iter = layers.erase(iter);
    }
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_OPTIMIZE_PASS(EliminateUnusefulNode);
}  // namespace TNN_CONVERTER
