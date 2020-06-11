// Copyright 2019 Tencent. All Rights Reserved

#include "tnn_impl_openvino.h"

#include <fstream>
#include "tnn/interpreter/abstract_model_interpreter.h"
#include "tnn/core/instance.h"
#include "tnn/interpreter/net_structure.h"

namespace TNN_NS {
    
TNNImplFactoryRegister<TNNImplFactory<TNNImplOpenVINO>>
    g_tnn_impl_openvino_factory_register(MODEL_TYPE_OPENVINO);

TNNImplOpenVINO::TNNImplOpenVINO() {}

TNNImplOpenVINO::~TNNImplOpenVINO() {}

Status TNNImplOpenVINO::Init(ModelConfig& config) {
    TNNImpl::Init(config);
    auto interpreter = CreateModelInterpreter(config.model_type);
    if (!interpreter) {
        return Status(TNNERR_NET_ERR, "interpreter is nil");
    }
    interpreter_ = std::shared_ptr<AbstractModelInterpreter>(interpreter);
    interpreter_->Interpret(config.params);
    return TNN_OK;
}

Status TNNImplOpenVINO::DeInit() {
    return TNN_OK;
}

Status TNNImplOpenVINO::AddOutput(const std::string& layer_name,
                                      int output_index) {

    return TNN_OK;
}

std::shared_ptr<Instance> TNNImplOpenVINO::CreateInst(
    NetworkConfig& net_config, Status& status, InputShapesMap inputs_shape) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, inputs_shape);
    return instance;
}

}  // namespace TNN_NS
