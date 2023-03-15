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

#include "tnn_optimize_pass.h"
namespace TNN_CONVERTER {

DECLARE_OPTIMIZE_PASS(FuseGLU);

std::string TnnOptimizeFuseGLUPass::PassName() {
    return "FuseGLU";
}

TNN_NS::Status TnnOptimizeFuseGLUPass::exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource) {
    auto& layers = net_structure.layers;

    // GLU = split + sigmoid + mul
    for (auto iter = layers.begin(); iter + 2 != layers.end(); iter++) {
        auto& split_layer = *iter;
        if (split_layer->type != TNN_NS::LAYER_SPLITV || split_layer->outputs.size() != 2) {
            continue;
        }

        auto sigmoid_iter  = iter + 1;
        auto mul_iter      = iter + 2;
        auto sigmoid_layer = *sigmoid_iter;
        auto mul_layer     = *mul_iter;
        if (sigmoid_layer->type != TNN_NS::LAYER_SIGMOID || mul_layer->type != TNN_NS::LAYER_MUL) {
            continue;
        }
        if (split_layer->outputs[1] != sigmoid_layer->inputs[0] || sigmoid_layer->outputs[0] != mul_layer->inputs[1]
            || split_layer->outputs[0] != mul_layer->inputs[0]) {
            continue;
        }

        auto* split_param  = dynamic_cast<TNN_NS::SplitVLayerParam*>(split_layer->param.get());
        if (split_param->slices.size() != 2 || split_param->slices[0] != split_param->slices[1]) {
            continue ;
        }

        auto glu_param        = new TNN_NS::GLULayerParam;
        glu_param->axis       = split_param->axis;
        split_layer->param    = std::shared_ptr<TNN_NS::LayerParam>(glu_param);
        split_layer->type     = TNN_NS::LAYER_GLU;
        split_layer->type_str = "GLU";

        split_layer->outputs.clear();
        split_layer->outputs = mul_layer->outputs;
        layers.erase(sigmoid_iter);
        mul_iter -= 1;
        layers.erase(mul_iter);
    }
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_OPTIMIZE_PASS(FuseGLU);
}  // namespace TNN_CONVERTER
