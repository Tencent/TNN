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

#include "tnn_optimize_pass.h"
namespace TNN_CONVERTER {

DECLARE_OPTIMIZE_PASS(EliminateSqueeze);

std::string TnnOptimizeEliminateSqueezePass::PassName() {
    return "EliminateSqueeze";
}

TNN_NS::Status TnnOptimizeEliminateSqueezePass::exec(tnn::NetStructure& net_structure, tnn::NetResource& net_resource) {
    auto& layers = net_structure.layers;
    for (auto iter = layers.begin(); iter != layers.end();) {
        auto& layer = *iter;
        if (layer->type != TNN_NS::LAYER_SQUEEZE) {
            iter++;
            continue;
        }
        // 处理悬空节点
        if (layer->inputs.empty() || layer->outputs.empty()) {
            iter = layers.erase(iter);
            continue;
        }
        // squeeze has only one input
        auto squeeze_input  = layer->inputs[0];
        auto squeeze_output = layer->outputs;
        for (auto sub_iter = layers.begin(); sub_iter != layers.end(); sub_iter++) {
            auto& sub_layer = *sub_iter;
            for (int i = 0; i < sub_layer->inputs.size(); i++) {
                if (std::find(squeeze_output.begin(), squeeze_output.end(), sub_layer->inputs[i]) !=
                    squeeze_output.end()) {
                    sub_layer->inputs[i] = squeeze_input;
                }
            }
        }
        iter = layers.erase(iter);
    }
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_OPTIMIZE_PASS(EliminateSqueeze);
}  // namespace TNN_CONVERTER