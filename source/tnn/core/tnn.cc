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

#include "tnn/core/tnn.h"

#include "tnn/core/tnn_impl.h"

namespace TNN_NS {

TNN::TNN() {}
TNN::~TNN() {
    DeInit();
}

Status TNN::Init(ModelConfig& config) {
    impl_ = TNNImplManager::GetTNNImpl(config.model_type);
    if (!impl_) {
        LOGE("Error: not support mode type: %d. If TNN is a static library, link it with option -Wl,--whole-archive tnn -Wl,--no-whole-archive on android or add -force_load on iOS\n", config.model_type);
        return Status(TNNERR_NET_ERR, "unsupported mode type, If TNN is a static library, link it with option -Wl,--whole-archive tnn -Wl,--no-whole-archive on android or add -force_load on iOS");
    }
    return impl_->Init(config);
}

Status TNN::DeInit() {
    impl_ = nullptr;
    return TNN_OK;
}

Status TNN::AddOutput(const std::string& layer_name, int output_index) {
    // todo for output index
    if (!impl_) {
        LOGE("Error: impl_ is nil\n");
        return Status(TNNERR_NET_ERR, "tnn impl_ is nil");
    }
    return impl_->AddOutput(layer_name, output_index);
}

Status TNN::GetModelInputShapesMap(InputShapesMap& shapes_map) {
     if (!impl_) {
        LOGE("Error: impl_ is nil\n");
        return Status(TNNERR_NET_ERR, "tnn impl_ is nil");
    }
    return impl_->GetModelInputShapesMap(shapes_map);
}

std::shared_ptr<Instance> TNN::CreateInst(NetworkConfig& config, Status& status, InputShapesMap inputs_shape) {
    if (!impl_) {
        status = Status(TNNERR_NET_ERR, "tnn impl_ is nil");
        return nullptr;
    }

    return impl_->CreateInst(config, status, inputs_shape);
}

std::shared_ptr<Instance> TNN::CreateInst(NetworkConfig& config, Status& status, InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape) {
    if (!impl_) {
        status = Status(TNNERR_NET_ERR, "tnn impl_ is nil");
        return nullptr;
    }

    return impl_->CreateInst(config, status, min_inputs_shape, max_inputs_shape);
}

}  // namespace TNN_NS
