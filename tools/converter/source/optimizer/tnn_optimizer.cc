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

#include "tnn_optimizer.h"

#include "tnn_optimize_pass.h"

namespace TNN_CONVERTER {

TNN_NS::Status TnnOptimizer::Optimize(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource) {
    std::vector<std::string> optimize_pass = {"EliminateUnusefulNode", "TransformReduceMean"};
    for (auto pass_name : optimize_pass) {
        auto pass = TnnOptimizePassManager::get()->search(pass_name);
        if (pass == nullptr) {
            LOGE("Unsupport optimize pass %s\n", pass_name.c_str());
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_PASS;
        }
        pass->exec(net_structure, net_resource);
    }
    return TNN_NS::TNN_CONVERT_OK;
}
}  // namespace TNN_CONVERTER
