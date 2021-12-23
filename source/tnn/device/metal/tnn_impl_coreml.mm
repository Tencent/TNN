#include "tnn/device/metal/tnn_impl_coreml.h"
#include "tnn/core/instance.h"
#include "tnn/interpreter/abstract_model_interpreter.h"
#include "tnn/interpreter/net_structure.h"
#include "stdio.h"

namespace TNN_NS {

TNNImplFactoryRegister<TNNImplFactory<TNNImplCoreML>>
    g_tnn_impl_coreml_factory_register(MODEL_TYPE_COREML);
    
TNNImplCoreML::TNNImplCoreML() {
    
}
    
TNNImplCoreML::~TNNImplCoreML() {
}

Status TNNImplCoreML::Init(ModelConfig& config) {
    return TNNImpl::Init(config);
}

Status TNNImplCoreML::DeInit() {
//    coreml_model_ = nil;
    return TNN_OK;
}

Status TNNImplCoreML::AddOutput(const std::string& layer_name, int output_index) {
    return Status(TNNERR_MODEL_ERR, "Error: CoreML do not support adding output");
}

Status TNNImplCoreML::GetModelInputShapesMap(InputShapesMap& shapes_map) {
    return Status(TNNERR_NET_ERR, "Error: CoreML do not supprt get model input shapes");
}

Status TNNImplCoreML::GetModelInputDataTypeMap(InputDataTypeMap& data_type_map) {
    return Status(TNNERR_NET_ERR, "Error: CoreML do not supprt get model input data types");
}

Status TNNImplCoreML::GetModelInputNames(std::vector<std::string>& input_names) {
    return Status(TNNERR_NET_ERR, "Error: CoreML do not supprt get model input names");
}

Status TNNImplCoreML::GetModelOutputNames(std::vector<std::string>& output_names) {
    return Status(TNNERR_NET_ERR, "Error: CoreML do not supprt get model output names");

}

std::shared_ptr<Instance> TNNImplCoreML::CreateInst(NetworkConfig& net_config,
                                               Status& status,
                                               InputShapesMap inputs_shape,
                                               InputDataTypeMap inputs_data_type) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(nullptr, inputs_shape, inputs_data_type);

    if (status != TNN_OK) {
        return nullptr;
    }
    return instance;
}


std::shared_ptr<Instance> TNNImplCoreML::CreateInst(NetworkConfig& net_config,
                                               Status& status,
                                               InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape,
                                               InputDataTypeMap inputs_data_type) {
    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(nullptr, min_inputs_shape, max_inputs_shape, inputs_data_type);

    if (status != TNN_OK) {
        return nullptr;
    }
    return instance;
}

    
}  // namespace TNN_NS
