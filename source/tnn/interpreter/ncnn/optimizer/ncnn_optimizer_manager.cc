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

#include "tnn/interpreter/ncnn/optimizer/ncnn_optimizer_manager.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tnn/interpreter/ncnn/optimizer/ncnn_optimizer.h"

namespace TNN_NS {

namespace ncnn {

    std::map<std::string, shared_ptr<NCNNOptimizer>> &NCNNOptimizerManager::GetNetOptimizerMap() {
        static std::map<std::string, std::shared_ptr<NCNNOptimizer>> s_ncnn_optimizer_map;
        return s_ncnn_optimizer_map;
    }

    std::vector<std::string> &NCNNOptimizerManager::GetNetOptimizerSeq() {
        static std::vector<std::string> s_ncnn_optimizer_seq;
        return s_ncnn_optimizer_seq;
    }

    Status NCNNOptimizerManager::Optimize(NetStructure *structure, NetResource *resource) {
        auto &optimizer_map  = NCNNOptimizerManager::GetNetOptimizerMap();
        auto &optimizer_list = NCNNOptimizerManager::GetNetOptimizerSeq();

        for (auto optimizer_name : optimizer_list) {
            auto optimizer = optimizer_map[optimizer_name];
            auto status    = optimizer->Optimize(structure, resource);
            if (status != TNN_OK) {
                return status;
            }
        }
        return TNN_OK;
    }

    void NCNNOptimizerManager::RegisterNetOptimizer(NCNNOptimizer *optimizer) {
        auto &optimizer_map           = NCNNOptimizerManager::GetNetOptimizerMap();
        auto optimizer_name           = optimizer->Strategy();
        optimizer_map[optimizer_name] = std::shared_ptr<NCNNOptimizer>(optimizer);
        NCNNOptimizerManager::GetNetOptimizerSeq().push_back(optimizer_name);
    }

}  // namespace ncnn

}  // namespace TNN_NS
