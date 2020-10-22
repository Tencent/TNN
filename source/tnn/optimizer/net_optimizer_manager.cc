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

#include "tnn/optimizer/net_optimizer_manager.h"

#include <algorithm>

namespace TNN_NS {

namespace optimizer {

    std::map<std::string, shared_ptr<NetOptimizer>> &NetOptimizerManager::GetNetOptimizerMap() {
        static std::map<std::string, std::shared_ptr<NetOptimizer>> s_net_optimizer_map;
        return s_net_optimizer_map;
    }

    std::vector<std::pair<OptPriority, std::string>> &NetOptimizerManager::GetNetOptimizerSeq() {
        static std::vector<std::pair<OptPriority, std::string>> s_net_optimizer_seq;
        return s_net_optimizer_seq;
    }

    Status NetOptimizerManager::Optimize(NetStructure *structure, NetResource *resource, const NetworkConfig &net_config) {
        auto &optimizer_map = NetOptimizerManager::GetNetOptimizerMap();
        std::sort(NetOptimizerManager::GetNetOptimizerSeq().begin(), NetOptimizerManager::GetNetOptimizerSeq().end());

        for (auto iter : NetOptimizerManager::GetNetOptimizerSeq()) {
            auto optimizer = optimizer_map[iter.second];
            if (optimizer->IsSupported(net_config)) {
                auto status = optimizer->Optimize(structure, resource);
                if (status != TNN_OK) {
                    return status;
                }
            }
        }

        return TNN_OK;
    }

    void NetOptimizerManager::RegisterNetOptimizer(NetOptimizer *optimizer, OptPriority prior) {
        if (optimizer && optimizer->Strategy().length() > 0) {
            auto &optimizer_map                  = NetOptimizerManager::GetNetOptimizerMap();
            optimizer_map[optimizer->Strategy()] = std::shared_ptr<NetOptimizer>(optimizer);
            NetOptimizerManager::GetNetOptimizerSeq().push_back(std::make_pair(prior, optimizer->Strategy()));
        }
    }

}  // namespace optimizer

}  // namespace TNN_NS
