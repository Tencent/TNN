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
    LOGE("Atlas not support this api (AddOutput)\n");
    return Status(TNNERR_DEVICE_NOT_SUPPORT, "Atlas not support this api (AddOutput)");
}

Status TNNImplAtlas::GetModelInputNames(std::vector<std::string>& input_names) {
    LOGE("Atlas not support this api (GetModelInputNames)\n");
    return Status(TNNERR_DEVICE_NOT_SUPPORT, "Atlas not support this api (GetModelInputNames)");
}

Status TNNImplAtlas::GetModelOutputNames(std::vector<std::string>& output_names) {
    LOGE("Atlas not support this api (GetModelOutputNames)\n");
    return Status(TNNERR_DEVICE_NOT_SUPPORT, "Atlas not support this api (GetModelOutputNames)");
}

Status TNNImplAtlas::GetModelInputShapesMap(InputShapesMap& shapes_map) {
    LOGE("Atlas not support this api (GetModelInputShapesMap)\n");
    return Status(TNNERR_DEVICE_NOT_SUPPORT, "Atlas not support this api (GetModelInputShapesMap)");
}

Status TNNImplAtlas::GetModelInputDataTypeMap(InputDataTypeMap& data_type_map) {
    LOGE("Atlas not support this api (GetModelInputDataTypeMap)\n");
    return Status(TNNERR_DEVICE_NOT_SUPPORT, "Atlas not support this api (GetModelInputDataTypeMap)");
}

std::shared_ptr<Instance> TNNImplAtlas::CreateInst(NetworkConfig& net_config, Status& status,
                                                   InputShapesMap inputs_shape, InputDataTypeMap inputs_data_type) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, inputs_shape, inputs_data_type);
    return instance;
}

std::shared_ptr<Instance> TNNImplAtlas::CreateInst(NetworkConfig& net_config, Status& status,
                                                   InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape, InputDataTypeMap inputs_data_type) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, min_inputs_shape, max_inputs_shape, inputs_data_type);
    return instance;
}

}  // namespace TNN_NS
