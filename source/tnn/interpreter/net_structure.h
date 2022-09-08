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

#ifndef TNN_SOURCE_TNN_INTERPRETER_NET_STRUCTURE_H_
#define TNN_SOURCE_TNN_INTERPRETER_NET_STRUCTURE_H_

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/utils/split_utils.h"

namespace TNN_NS {

// @brief LayerInfo describes layer name, type, inputs, outputs and
// parameter info
struct LayerInfo {
    LayerType type;
    std::string type_str;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::shared_ptr<LayerParam> param = nullptr;

public:
    std::shared_ptr<LayerInfo> Copy() {
        std::shared_ptr<LayerInfo> layer_info(new LayerInfo());
        if (nullptr == layer_info) {
            return nullptr;
        }
        *layer_info.get() = *this;
        if (param) {
            auto layer_param = param->Copy();
            if (nullptr != layer_param) {
                layer_info->param = layer_param;
            }
        }

        return layer_info;
    }
};

// @brief NetStructure describes network build info
struct NetStructure {
    InputShapesMap inputs_shape_map;
    InputDataTypeMap  input_data_type_map;
    std::set<std::string> outputs;
    std::vector<std::shared_ptr<LayerInfo>> layers;
    std::set<std::string> blobs;
    ModelType source_model_type = MODEL_TYPE_TNN;

public:
    std::shared_ptr<NetStructure> Copy() {
        std::shared_ptr<NetStructure> net_structure(new NetStructure());
        if (nullptr == net_structure) {
            return nullptr;
        }
        *net_structure.get() = *this;
        net_structure->layers.clear();
        for (auto layer_info : layers) {
            auto layer_info_new = layer_info->Copy();
            if (nullptr != layer_info_new) {
                net_structure->layers.push_back(layer_info_new);
            } else {
                net_structure->layers.push_back(layer_info);
            }
        }
        return net_structure;
    }

    NetStructure* CreateNew() {
        NetStructure* net_structure(new NetStructure());
        if (nullptr == net_structure) {
            return nullptr;
        }
        *net_structure = *this;
        net_structure->layers.clear();
        for (auto layer_info : layers) {
            auto layer_info_new = layer_info->Copy();
            if (nullptr != layer_info_new) {
                net_structure->layers.push_back(layer_info_new);
            } else {
                net_structure->layers.push_back(layer_info);
            }
        }
        return net_structure;
    }
};

std::shared_ptr<LayerInfo> GetLayerInfoFromName(NetStructure* net_struct, std::string name);

bool GetQuantizedInfoFromNetStructure(NetStructure* net_struct);

bool NeedDoConstantFolding(NetStructure* net_struct);

bool IsQuantizedLayerFromInputName(NetStructure* net_structure, const std::string& input_name);
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_NET_STRUCTURE_H_
