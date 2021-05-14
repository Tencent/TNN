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

DECLARE_OPTIMIZE_PASS(ConstantFolding);

std::string TnnOptimizeConstantFoldingPass::PassName() {
    return "ConstantFolding";
}

TNN_NS::Status TnnOptimizeConstantFoldingPass::exec(TNN_NS::NetStructure& net_structure,
                                                    TNN_NS::NetResource& net_resource) {
    std::set<TNN_NS::LayerType> black_list = {TNN_NS::LAYER_SHAPE};

    auto& constant_map    = net_resource.constant_map;
    auto& constant_layers = net_resource.constant_layers;
    auto& net_layers      = net_structure.layers;
    for (auto iter = net_layers.begin(); iter != net_layers.end();) {
        auto& layer = *iter;
        if (constant_layers.find(layer->name) == constant_layers.end()) {
            iter++;
            continue;
        }
        auto layer_output_names   = layer->outputs;
        const auto& pre_node_name = layer->inputs[0];
        for (auto sub_iter = iter; sub_iter < net_layers.end(); sub_iter++) {
            auto sub_layer = *sub_iter;
            for (auto& input_name : sub_layer->inputs) {
                if (std::find(layer_output_names.begin(), layer_output_names.end(), input_name) !=
                    layer_output_names.end()) {
                    input_name = pre_node_name;
                }
            }
        }
        iter = net_layers.erase(iter);
    }
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_OPTIMIZE_PASS(ConstantFolding);
}  // namespace TNN_CONVERTER
