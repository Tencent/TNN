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

#include "tnn/optimizer/net_optimizer_remove_layers.h"

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "tnn/core/common.h"
#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"

namespace TNN_NS {

namespace optimizer {

    static std::set<LayerType> global_remove_layer_types_set = {
        LAYER_SPLITING,
        LAYER_FLATTEN,
        LAYER_DROPOUT,
        LAYER_CONCAT,
    };

    // P1 priority: should be fuse after bn scale fuse
    NetOptimizerRegister<NetOptimizerRemoveLayers> g_net_optimizer_remove_split(OptPriority::P1);

    std::string NetOptimizerRemoveLayers::Strategy() {
        return kNetOptimizerRemoveLayers;
    }

    bool NetOptimizerRemoveLayers::IsSupported(const NetworkConfig &net_config) {
        return true;
    }

    Status NetOptimizerRemoveLayers::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        if (structure->source_model_type != MODEL_TYPE_NCNN) {
            return TNN_OK;
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();
        if (count <= 1) {
            return TNN_OK;
        }

        auto &ly_sets = global_remove_layer_types_set;

        std::vector<std::shared_ptr<LayerInfo>> layers_fused;

        std::map<std::string, std::string> rename_map;
        for (int index = 0; index < count; index++) {
            auto layer = layers_orig[index];
            if (ly_sets.find(layer->type) != ly_sets.end() && layer->inputs.size() == 1) {
                auto in_name = layer->inputs[0];
                for (auto out_name : layer->outputs) {
                    if (rename_map.find(out_name) == rename_map.end()) {
                        rename_map[out_name] = in_name;
                    } else {
                        return Status(TNNERR_NET_ERR, "duplicated output blobs");
                    }
                }
            } else {
                std::vector<std::string> new_inputs;
                new_inputs.reserve(layer->inputs.size());
                for (auto in_name : layer->inputs) {
                    while (rename_map.find(in_name) != rename_map.end()) {
                        in_name = rename_map[in_name];
                    }
                    new_inputs.push_back(in_name);
                }
                layer->inputs = new_inputs;
                layers_fused.push_back(layer);
            }
        }

        structure->layers = layers_fused;

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
