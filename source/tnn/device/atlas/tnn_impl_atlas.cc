// Copyright 2019 Tencent. All Rights Reserved

#include "tnn_impl_atlas.h"
#include <fstream>
#include "tnn/core/instance.h"
#include "tnn/interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

TNNImplFactoryRegister<TNNImplFactory<TNNImplAtlas>>
    g_tnn_impl_atlas_factory_register(MODEL_TYPE_ATLAS);

TNNImplAtlas::TNNImplAtlas() {}

TNNImplAtlas::~TNNImplAtlas() {}

Status TNNImplAtlas::Init(ModelConfig& config) {
    TNNImpl::Init(config);
    auto interpreter = CreateModelInterpreter(config.model_type);
    if (!interpreter) {
        return Status(TNNERR_NET_ERR, "interpreter is nil");
    }
    interpreter_ = std::shared_ptr<AbstractModelInterpreter>(interpreter);
    interpreter_->Interpret(config.params);
    return TNN_OK;
}

Status TNNImplAtlas::DeInit() {
    return TNN_OK;
}

Status TNNImplAtlas::AddOutput(const std::string& layer_name,
                               int output_index) {
    return TNN_OK;
}

std::shared_ptr<Instance> TNNImplAtlas::CreateInst(
    NetworkConfig& net_config, Status& status, InputShapesMap inputs_shape) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, inputs_shape);
    return instance;
}

}  // namespace TNN_NS
