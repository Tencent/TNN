// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <fstream>

#include "tnn/core/instance.h"
#include "tnn/device/snpe/tnn_impl_snpe.h"
#include "tnn/interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

TNNImplFactoryRegister<TNNImplFactory<TNNImplSnpe>>
    g_tnn_impl_snpe_factory_register(MODEL_TYPE_SNPE);

TNNImplSnpe::TNNImplSnpe() {}

TNNImplSnpe::~TNNImplSnpe() {}

Status TNNImplSnpe::Init(ModelConfig& config) {
    TNNImpl::Init(config);
    auto interpreter = CreateModelInterpreter(config.model_type);
    if (!interpreter) {
        return Status(TNNERR_NET_ERR, "interpreter is nil");
    }
    interpreter_ = std::shared_ptr<AbstractModelInterpreter>(interpreter);
    interpreter_->Interpret(config.params);
    return TNN_OK;
}

Status TNNImplSnpe::DeInit() {
    return TNN_OK;
}

Status TNNImplSnpe::AddOutput(const std::string& layer_name,
                              int output_index) {
    return TNN_OK;
}

//Status TNNImplSnpe::AddOutput(const std::string& layer_name, int output_index) {
//    return Status(TNNERR_MODEL_ERR, "Error: TNN SNPE Impl does not support adding output");
//}

Status TNNImplSnpe::GetModelInputShapesMap(InputShapesMap& shapes_map) {
    return Status(TNNERR_NET_ERR, "Error: TNN SNPE Impl does not supprt get model input shapes");
}

Status TNNImplSnpe::GetModelInputDataTypeMap(InputDataTypeMap& data_type_map) {
    return Status(TNNERR_NET_ERR, "Error: TNN SNPE Impl does not supprt get model input data types");
}

Status TNNImplSnpe::GetModelInputNames(std::vector<std::string>& input_names) {
    return Status(TNNERR_NET_ERR, "Error: TNN SNPE Impl does not supprt get model input names");
}

Status TNNImplSnpe::GetModelOutputNames(std::vector<std::string>& output_names) {
    return Status(TNNERR_NET_ERR, "Error: TNN SNPE Impl does not supprt get model output names");
}

std::shared_ptr<Instance> TNNImplSnpe::CreateInst(NetworkConfig& net_config,
                                                  Status& status,
                                                  InputShapesMap inputs_shape,
                                                  InputDataTypeMap inputs_data_type) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, inputs_shape, inputs_data_type);
    return instance;
}

std::shared_ptr<Instance> TNNImplSnpe::CreateInst(NetworkConfig& net_config,
                                                  Status& status,
                                                  InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape,
                                                  InputDataTypeMap inputs_data_type) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, min_inputs_shape, max_inputs_shape, inputs_data_type);
    return instance;
}

}  // namespace TNN_NS
