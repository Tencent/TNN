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

#include "tnn/optimizer/net_optimizer_config_layer_param.h"

#include <map>
#include <memory>
#include <vector>
#include <sstream>
#include <iostream>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"

namespace TNN_NS {

namespace optimizer {

    // P1 priority: should be fuse after bn scale fuse
    NetOptimizerRegister<NetOptimizerConfigLayerParam> g_net_optimizer_config_layer_param(OptPriority::P2);

    std::string NetOptimizerConfigLayerParam::Strategy() {
        return kNetOptimizerConfigLayerParam;
    }

    bool NetOptimizerConfigLayerParam::IsSupported(const NetworkConfig &net_config) {
        if (net_config.layer_param_config.empty()) {
            return false;
        } else {
            this->layer_param_map_ptr = 
                std::make_shared<std::map<std::string, std::string>>(
                net_config.layer_param_config);
            return true;
        }
    }

    Status NetOptimizerConfigLayerParam::Optimize(NetStructure *structure, NetResource *resource) {
        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();

        for (int index = 0; index < count; index++) {
            auto layer_info = layers_orig[index];
            auto layer_param = layer_info->param.get();

            auto layer_search = this->layer_param_map_ptr->find(layer_info->name);
            if (layer_search != this->layer_param_map_ptr->end()) {
                auto config_str = layer_search->second;
                // config_str format is [key1:value1,key2:value2]
                // store this string to map<str, str>
                std::stringstream ss(config_str);
                while (ss.good()) {
                    std::string sub_str;
                    getline(ss, sub_str, ',');
                    auto split_pos = sub_str.find(':');
                    if (split_pos != std::string::npos) {
                        layer_param->extra_config.emplace(
                            sub_str.substr(0, split_pos), sub_str.substr(split_pos + 1));
                    }
                }
            }
        }

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS

