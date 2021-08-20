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

// author: sanerzheng@tencent.com

#ifndef TNN_SOURCE_TNN_TRAIN_BASE_SOLVER_SOLVER_H
#define TNN_SOURCE_TNN_TRAIN_BASE_SOLVER_SOLVER_H
#include <string>
#include <set>

#include "tnn/core/status.h"
#include "tnn/core/blob.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/train/grad/grad_manager.h"
#include "tnn/train/grad/train_context.h"
namespace TNN_NS {
namespace train {
class BaseSolver{
public:
    BaseSolver(AbstractNetwork* network, NetworkConfig* config) {
        auto& context = grad_manager_.GetContext();
        context.network = network;
        context.config = config;
    };
     ~BaseSolver() {};
     Status step();
    // int CurrentStep() {
    //     return step_;
    // };
    // void SetCurrentStep(int step) {
    //     step_ = step;
    // };
     void SetNeedGradLayers(const std::set<std::string>& need_grad_layers);
public:
    // @brief 更新参数的梯度值，按现在的框架只有resource里的资源需要做变量的更新
    virtual Status UpdateTrainableVariable(RawBuffer* resource_param, const std::shared_ptr<RawBuffer>& resource_param_grad);
    virtual Status ComputeUpdateValue(RawBuffer* resource_param, std::shared_ptr<RawBuffer>& resource_param_grad);

    GradManager grad_manager_;
private:
    int step_ = 0;
    
    
};

} // namespace train
} // namespace TNN_NS
#endif  // TNN_SOURCE_TNN_TRAIN_BASE_SOLVER_SOLVER_H