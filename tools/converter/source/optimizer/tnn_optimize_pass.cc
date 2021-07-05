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

TnnOptimizePassManager* TnnOptimizePassManager::tnn_optimize_pass_manager_ = nullptr;

TnnOptimizePassManager::~TnnOptimizePassManager() {
    for (auto& iter : tnn_optimize_pass_map_) {
        delete iter.second;
    }
    tnn_optimize_pass_map_.clear();
}

TnnOptimizePassManager* TnnOptimizePassManager::get() {
    if (tnn_optimize_pass_manager_ == nullptr) {
        tnn_optimize_pass_manager_ = new TnnOptimizePassManager;
    }
    return tnn_optimize_pass_manager_;
}

TnnOptimizePass* TnnOptimizePassManager::search(const std::string pass_name) {
    auto iter = tnn_optimize_pass_map_.find(pass_name);
    if (iter == tnn_optimize_pass_map_.end()) {
        return nullptr;
    }
    return iter->second;
}

void TnnOptimizePassManager::insert(const std::string pass_name, TnnOptimizePass* t) {
    tnn_optimize_pass_map_.insert(std::make_pair(pass_name, t));
}
}  // namespace TNN_CONVERTER
