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

#ifndef TNN_SOURCE_TNN_TRAIN_SOLVER_BASE_SOLVER_H
#define TNN_SOURCE_TNN_TRAIN_SOLVER_BASE_SOLVER_H

#include <set>
#include <string>

#include "tnn/core/blob.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/train/grad/grad_manager.h"
#include "tnn/train/grad/train_context.h"

namespace TNN_NS {
namespace train {

    class BaseSolver {
    public:
        BaseSolver(AbstractNetwork *network, NetworkConfig *config);

        ~BaseSolver();

        virtual Status Step();

        void SetNeedGradLayers(const std::set<std::string> &need_grad_layers);

    private:
        // @brief Update the gradient by learning rate, etc. support different strategies
        virtual Status ComputeUpdateValue(RawBuffer *param, std::shared_ptr<RawBuffer> &grad) = 0;

        // @brief Update the param by update_value
        virtual Status UpdateTrainableVariable(RawBuffer *param, const std::shared_ptr<RawBuffer> &update_value);

    private:
        GradManager grad_manager_;
    };

}  // namespace train
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_TRAIN_SOLVER_BASE_SOLVER_H
