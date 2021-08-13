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

#ifndef TNN_SOURCE_TNN_TRAIN_GRAD_GRAD_H
#define TNN_SOURCE_TNN_TRAIN_GRAD_GRAD_H

#include <string>
#include <set>
#include "core/tnn.h"
#include "tnn/train/grad/layer_grad.h"
#include "tnn/core/default_network.h"
#include "tnn/train/grad/train_context.h"

namespace TNN_NS {
namespace train {
class GradManager {
public:
    GradManager(){};
    GradManager(AbstractNetwork* network, NetworkConfig* config);
    ~GradManager() = default;
    Status CalcuteGrads();
    inline TrainContext& GetContext();
    void SetNeedGradLayers(const std::set<std::string>& trainable_layers);
private:
    //static std::set<RawBuffer* > trainables_;
    TrainContext context_;
    std::set<std::string> need_grad_layers_;
};



} // namespace train
} // namespace TNN_NS
#endif  // TNN_SOURCE_TNN_TRAIN_GRAD_GRAD_H