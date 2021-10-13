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

#include "tnn/interpreter/net_structure.h"

#include <algorithm>

namespace TNN_NS {

std::shared_ptr<LayerInfo> GetLayerInfoFromName(NetStructure* net_struct, std::string name) {
    std::shared_ptr<LayerInfo> layer_info;
    for (auto item : net_struct->layers) {
        if (item != nullptr && item->name == name) {
            layer_info = item;
            break;
        }
    }

    return layer_info;
}

bool GetQuantizedInfoFromNetStructure(NetStructure* net_struct) {
    std::vector<std::shared_ptr<LayerInfo>> layers = net_struct->layers;
    auto quantize_layer = std::find_if(layers.begin(), layers.end(), [](std::shared_ptr<LayerInfo> iter) {
        return iter->param->quantized == true;
    });
    return quantize_layer != layers.end();
}

bool NeedDoConstantFolding(NetStructure* net_struct) {
    if (!net_struct) {
        return false;
    }
    
    for (auto item : net_struct->layers) {
        if (item != nullptr &&
            (item->type == LAYER_SHAPE || item->type_str == "Shape")) {
            return true;
        }
    }
    
    return false;
}

bool IsQuantizedLayerFromInputName(NetStructure* net_structure, const std::string& input_name) {
    for (const auto& layer_info : net_structure->layers) {
        const auto& inputs = layer_info->inputs;
        if (std::find(inputs.begin(), inputs.end(), input_name) != inputs.end()) {
            return layer_info->param->quantized;
        }
    }
    return false;
}

}  // namespace TNN_NS
