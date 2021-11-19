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

#ifndef TNN_SOURCE_TNN_TRAIN_SOLVER_SOLVER_STRATEGY_H_
#define TNN_SOURCE_TNN_TRAIN_SOLVER_SOLVER_STRATEGY_H_

#include <map>
#include <string>

#include "tnn/core/blob.h"
#include "tnn/core/status.h"
#include "tnn/layer/base_layer.h"
#include "tnn/train/training_info.h"

namespace TNN_NS {

class SolverStrategy {
public:
    SolverStrategy();

    virtual ~SolverStrategy();

    virtual Status OnSolve(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs, SolverParam *param,
                           Context *context, const SolverStrategyInfo &solver_info);

    static Status RegisterSolverStrategy(DeviceType device, SolverType type,
                                         std::shared_ptr<SolverStrategy> solver_strategy);

    static SolverStrategy *GetSolverStrategy(DeviceType device, SolverType type);

private:
    static std::map<std::pair<DeviceType, SolverType>, std::shared_ptr<SolverStrategy>> &GetSolverStrategyMap();

    virtual Status ExecUpdate(Blob *grad, RawBuffer *resource, SolverParam *param, Context *context) = 0;
};

template <typename T>
class SolverStrategyRegister {
public:
    explicit SolverStrategyRegister(DeviceType device, SolverType type) {
        SolverStrategy::RegisterSolverStrategy(device, type, std::make_shared<T>());
    }
};

#define DECLARE_SOLVER_STRATEGY(device_string, device, type_string, solver_type)                                       \
    class device_string##type_string##SolverStrategy : public SolverStrategy {                                         \
    public:                                                                                                            \
        virtual ~device_string##type_string##SolverStrategy(){};                                                       \
        virtual Status ExecUpdate(Blob *grad, RawBuffer *resource, SolverParam *param, Context *context);              \
    };

#define REGISTER_SOLVER_STRATEGY(device_string, device, type_string, solver_type)                                      \
    SolverStrategyRegister<device_string##type_string##SolverStrategy>                                                 \
        g_##device##_##solver_type##_solver_strategy_register(device, solver_type);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_TRAIN_SOLVER_SOLVER_STRATEGY_H_
