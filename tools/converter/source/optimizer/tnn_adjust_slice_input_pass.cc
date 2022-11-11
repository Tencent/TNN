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

DECLARE_OPTIMIZE_PASS(AdjustSliceInput);

std::string TnnOptimizeAdjustSliceInputPass::PassName() {
    return "AdjustSliceInput";
}

TNN_NS::Status TnnOptimizeAdjustSliceInputPass::exec(TNN_NS::NetStructure& net_structure,
                                                     TNN_NS::NetResource& net_resource) {
    std::set<TNN_NS::LayerType> black_list = {TNN_NS::LAYER_STRIDED_SLICE_V2};

    auto& constant_map    = net_resource.constant_map;
    auto& constant_layers = net_resource.constant_layers;
    auto& net_layers      = net_structure.layers;
    for (auto& iter : net_layers) {
        auto& layer = *iter;
        if (black_list.find(layer.type) == black_list.end()) {
            continue;
        }
        auto& inputs = layer.inputs;
        for (int i = 1; i < inputs.size(); ++i) {
            const auto& input_name = inputs[i];
            if (constant_map.find(input_name) == constant_map.end() &&
                constant_layers.find(input_name) == constant_layers.end()) {
                LOGE("AdjustSliceInput: Stride_Slice_v2 get invalid dynamic input\n");
                return TNN_NS::TNNERR_CONVERT_OPTIMIZE_ERROR;
            }
        }
        // stride_slice_v2 has only one input
        layer.inputs = {inputs[0]};
    }
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_OPTIMIZE_PASS(AdjustSliceInput);
}  // namespace TNN_CONVERTER
