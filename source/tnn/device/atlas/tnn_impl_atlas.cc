// Copyright 2019 Tencent. All Rights Reserved

#include "tnn_impl_atlas.h"
#include <fstream>
#include "tnn/core/instance.h"
#include "tnn/interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

TNNImplFactoryRegister<TNNImplFactory<TNNImplAtlas>> g_tnn_impl_atlas_factory_register(MODEL_TYPE_ATLAS);

TNNImplAtlas::TNNImplAtlas() {}

TNNImplAtlas::~TNNImplAtlas() {}

Status TNNImplAtlas::Init(ModelConfig& config) {
    TNNImpl::Init(config);
    auto interpreter = CreateModelInterpreter(config.model_type);
    if (!interpreter) {
        return Status(TNNERR_NET_ERR, "interpreter is nil");
    }
    interpreter_ = std::shared_ptr<AbstractModelInterpreter>(interpreter);
    return interpreter_->Interpret(config.params);
}

Status TNNImplAtlas::DeInit() {
    return TNN_OK;
}

Status TNNImplAtlas::AddOutput(const std::string& layer_name, int output_index) {
    return TNN_OK;
}

Status TNNImplAtlas::GetModelInputShapesMap(InputShapesMap& shapes_map) {
    LOGE("Atlas not support this api (GetModelInputShapesMap)\n");
    return Status(TNNERR_DEVICE_NOT_SUPPORT, "Atlas not support this api (GetModelInputShapesMap)");
}

std::shared_ptr<Instance> TNNImplAtlas::CreateInst(NetworkConfig& net_config, Status& status,
                                                   InputShapesMap inputs_shape) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, inputs_shape);
    return instance;
}

std::shared_ptr<Instance> TNNImplAtlas::CreateInst(NetworkConfig& net_config, Status& status,
                                                   InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, min_inputs_shape, max_inputs_shape);
    return instance;
}

}  // namespace TNN_NS
