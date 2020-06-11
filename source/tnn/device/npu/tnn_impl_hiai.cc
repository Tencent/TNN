// Copyright 2019 Tencent. All Rights Reserved

#include "tnn_impl_hiai.h"
#include <fstream>
#include "core/instance.h"
#include "interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

TNNImplFactoryRegister<TNNImplFactory<TNNImplHiai>>
    g_tnn_impl_hiai_factory_register(MODEL_TYPE_HIAI);

TNNImplHiai::TNNImplHiai() {}

TNNImplHiai::~TNNImplHiai() {}

Status TNNImplHiai::Init(ModelConfig& config) {
    TNNImpl::Init(config);
    auto interpreter = CreateModelInterpreter(config.model_type);
    if (!interpreter) {
        return Status(TNNERR_NET_ERR, "interpreter is nil");
    }
    interpreter_ = std::shared_ptr<AbstractModelInterpreter>(interpreter);
    interpreter_->Interpret(config.params);
    return TNN_OK;
}

Status TNNImplHiai::DeInit() {
    return TNN_OK;
}

Status TNNImplHiai::AddOutput(const std::string& layer_name,
                                   int output_index) {
    return TNN_OK;
}

std::shared_ptr<Instance> TNNImplHiai::CreateInst(
    NetworkConfig& net_config, Status& status, InputShapesMap inputs_shape) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, inputs_shape);
    return instance;
}

}  // namespace TNN_NS
