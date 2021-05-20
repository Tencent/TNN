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

#include "tnn/core/tnn_impl.h"

#include "tnn/interpreter/net_structure.h"

namespace TNN_NS {

TNNImpl::~TNNImpl() {
    DeInit();
}

Status TNNImpl::DeInit() {
    return TNN_OK;
}

Status TNNImpl::Init(ModelConfig &config) {
    model_config_.model_type = config.model_type;
    return TNN_OK;
}

std::map<ModelType, std::shared_ptr<AbstractTNNImplFactory>> &TNNImplManager::GetTNNImplFactoryMap() {
    static std::map<ModelType, std::shared_ptr<AbstractTNNImplFactory>> s_tnn_impl_factory_map;
    return s_tnn_impl_factory_map;
}

std::shared_ptr<TNNImpl> TNNImplManager::GetTNNImpl(ModelType type) {
    auto &impl_map = TNNImplManager::GetTNNImplFactoryMap();
    auto iter      = impl_map.find(type);
    if (iter != impl_map.end()) {
        return iter->second->CreateTNNImp();
    }

    return nullptr;
}

void TNNImplManager::RegisterTNNImplFactory(ModelType type, AbstractTNNImplFactory *factory) {
    if (factory) {
        auto &optimizer_map = TNNImplManager::GetTNNImplFactoryMap();
        optimizer_map[type] = std::shared_ptr<AbstractTNNImplFactory>(factory);
    }
}

}  // namespace TNN_NS
