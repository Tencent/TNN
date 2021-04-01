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

DECLARE_OPTIMIZE_PASS(FuseShuffleChannel);

std::string TnnOptimizeFuseShuffleChannelPass::PassName() {
    return "FuseShuffleChannel";
}

TNN_NS::Status TnnOptimizeFuseShuffleChannelPass::exec(TNN_NS::NetStructure& net_structure,
                                                       TNN_NS::NetResource& net_resource) {
    auto& layers = net_structure.layers;

    // ShuffleChannel <= Reshape - Transpose - Reshape
    for (auto iter = layers.begin(); iter + 2 != layers.end(); iter++) {
        auto& reshape_layer1 = *iter;
        if (reshape_layer1->type != TNN_NS::LAYER_RESHAPE || reshape_layer1->outputs.size() != 1) {
            continue;
        }

        auto transpose_iter = iter + 1;
        auto reshape_iter2  = iter + 2;
        auto tranpose_layer = *transpose_iter;
        auto reshape_layer2 = *reshape_iter2;
        if (tranpose_layer->type != TNN_NS::LAYER_PERMUTE || reshape_layer2->type != TNN_NS::LAYER_RESHAPE) {
            continue;
        }
        if (tranpose_layer->outputs.size() != 1) {
            continue;
        }
        if (reshape_layer1->outputs[0] != tranpose_layer->inputs[0] ||
            tranpose_layer->outputs[0] != reshape_layer2->inputs[0]) {
            continue;
        }

        auto* reshape_param1  = dynamic_cast<TNN_NS::ReshapeLayerParam*>(reshape_layer1->param.get());
        auto* transpose_param = dynamic_cast<TNN_NS::PermuteLayerParam*>(tranpose_layer->param.get());
        auto* reshape_param2  = dynamic_cast<TNN_NS::ReshapeLayerParam*>(reshape_layer2->param.get());
        const auto shape1     = reshape_param1->shape;
        const auto perm       = transpose_param->orders;
        const auto shape3     = reshape_param2->shape;

        int64_t group = 0;

        if (shape1.size() == 5 && perm.size() == 5) {
            // batch groups channels_per_group, height, width
            group = shape1[1];

            // 0 2 1 3 4
            if (perm[0] != 0 || perm[1] != 2 || perm[2] != 1 || perm[3] != 3 || perm[4] != 4) {
                continue;
            }

            if (shape3.size() != 4 || shape3[0] != shape1[0] ||
                (shape3[1] != -1 && shape3[1] != shape1[1] * shape1[2]) || shape3[2] != shape1[3] ||
                shape3[3] != shape1[4]) {
                continue;
            }
        } else if (shape1.size() == 3 && perm.size() == 3) {
            // groups, channels_per_group, height*width
            group = shape1[0];
            // 1 0 2
            if (perm[0] != 1 || perm[1] != 0 || perm[2] != 2) {
                continue;
            }

            // TODO：考虑情况shape3各种大小
            if (shape3.size() != 5 || shape3[0] != shape1[0] ||
                (shape3[1] != -1 && shape3[1] != shape1[1] * shape1[2]) || shape3[2] != shape1[3]) {
                continue;
            }
        } else {
            continue;
        }

        auto shuffle_param       = new TNN_NS::ShuffleLayerParam;
        shuffle_param->group     = group;
        reshape_layer1->param    = std::shared_ptr<TNN_NS::LayerParam>(shuffle_param);
        reshape_layer1->type     = TNN_NS::LAYER_SHUFFLE_CHANNEL;
        reshape_layer1->type_str = "ShuffleChannel";

        reshape_layer1->outputs.clear();
        reshape_layer1->outputs = reshape_layer2->outputs;
        layers.erase(transpose_iter);
        reshape_iter2 -= 1;
        layers.erase(reshape_iter2);
    }
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_OPTIMIZE_PASS(FuseShuffleChannel);
}  // namespace TNN_CONVERTER
