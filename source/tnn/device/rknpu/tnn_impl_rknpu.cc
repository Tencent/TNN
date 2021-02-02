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

#include "tnn/device/rknpu/tnn_impl_rknpu.h"
#include "tnn/core/instance.h"
#include "tnn/interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

TNNImplRknpu::TNNImplRknpu() {}

TNNImplRknpu::~TNNImplRknpu() {}

Status TNNImplRknpu::Init(ModelConfig& config) {
    TNNImpl::Init(config);
    auto interpreter = CreateModelInterpreter(config.model_type);
    if (!interpreter) {
        return Status(TNNERR_NET_ERR, "interpreter is nil");
    }
    interpreter_ = std::shared_ptr<AbstractModelInterpreter>(interpreter);
    return interpreter_->Interpret(config.params);
}

Status TNNImplRknpu::DeInit() {
    return TNN_OK;
}

Status TNNImplRknpu::AddOutput(const std::string& layer_name, int output_index) {
    return TNN_OK;
}

std::shared_ptr<Instance> TNNImplRknpu::CreateInst(NetworkConfig& net_config, Status& status,
                                                   InputShapesMap inputs_shape) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, inputs_shape);
    return instance;
}

std::shared_ptr<Instance> TNNImplRknpu::CreateInst(NetworkConfig& net_config, Status& status,
                                                   InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, min_inputs_shape, max_inputs_shape);
    return instance;
}

TNNImplFactoryRegister<TNNImplFactory<TNNImplRknpu>> g_tnn_impl_atlas_factory_register(MODEL_TYPE_RKCACHE);

}  // namespace TNN_NS
