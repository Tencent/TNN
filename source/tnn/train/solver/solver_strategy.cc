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

#include "tnn/train/solver/solver_strategy.h"

namespace TNN_NS {

SolverStrategy::SolverStrategy() {}

SolverStrategy::~SolverStrategy() {}

Status SolverStrategy::OnSolve(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                               SolverParam *param, Context *context, const SolverStrategyInfo &solver_info) {
    auto &trainables = solver_info.trainable_resources;
    // the last input is global step init value
    if (inputs.size() < 1) {
        LOGE("SolverStrategy::OnSolve, ERROR, input size error, should contain global step init value\n");
        return Status(TNNERR_TRAIN_ERROR, "input size error");
    }
    if (trainables.size() != (inputs.size() - 1)) {
        LOGE("SolverStrategy::OnSolve, ERROR, grad and resource count not equal\n");
        return Status(TNNERR_TRAIN_ERROR, "grad and resource count not equal");
    }

    for (int i = 0; i < inputs.size() - 1; ++i) {
        CHECK_PARAM_NULL(inputs[i]);
        CHECK_PARAM_NULL(trainables[i]);
        LOGD("Update: [%d] %s -> %d\n", i, inputs[i]->GetBlobDesc().description().c_str(),
             trainables[i]->GetDataCount());
        if (DimsVectorUtils::Count(inputs[i]->GetBlobDesc().dims) != trainables[i]->GetDataCount()) {
            LOGE("SolverStrategy::DoForward ERROR, grad and param data count not equal\n");
            return Status(TNNERR_TRAIN_ERROR, "grad and param data count not equal");
        }
        RETURN_ON_NEQ(ExecUpdate(inputs[i], trainables[i], param, context), TNN_OK);
    }

    return TNN_OK;
}

Status SolverStrategy::RegisterSolverStrategy(DeviceType device, SolverType type,
                                              std::shared_ptr<SolverStrategy> solver_strategy) {
    GetSolverStrategyMap()[{device, type}] = solver_strategy;
    return TNN_OK;
}

SolverStrategy *SolverStrategy::GetSolverStrategy(DeviceType device, SolverType type) {
    auto &solver_strategy_map = GetSolverStrategyMap();
    if (solver_strategy_map.count({device, type}) > 0) {
        return solver_strategy_map[{device, type}].get();
    }
    return nullptr;
}

std::map<std::pair<DeviceType, SolverType>, std::shared_ptr<SolverStrategy>> &SolverStrategy::GetSolverStrategyMap() {
    static std::map<std::pair<DeviceType, SolverType>, std::shared_ptr<SolverStrategy>> solver_strategy_map;
    return solver_strategy_map;
}

}  // namespace TNN_NS
