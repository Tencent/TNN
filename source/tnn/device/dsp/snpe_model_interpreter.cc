// Copyright 2019 Tencent. All Rights Reserved

#include "snpe_model_interpreter.h"
#include <fstream>

namespace TNN_NS {

SnpeModelInterpreter::SnpeModelInterpreter() {}

SnpeModelInterpreter::~SnpeModelInterpreter() {}

Status SnpeModelInterpreter::Interpret(std::vector<std::string> params) {
    std::string dlc_content = params[0];
    std::ifstream dlc_file(dlc_content);
    if (!dlc_file) {
        LOGE("Invalied dlc file path!\n");
        return TNNERR_INVALID_MODEL;
    }

    container_ = zdl::DlContainer::IDlContainer::open(
        zdl::DlSystem::String(dlc_content.c_str()));
    if (container_ == nullptr) {
        LOGE("Load dlc file failed!\n");
        return TNNERR_INVALID_MODEL;
    }

    return TNN_OK;
}

std::unique_ptr<zdl::DlContainer::IDlContainer>&
SnpeModelInterpreter::GetContainer() {
    return container_;
}

TypeModelInterpreterRegister<TypeModelInterpreterCreator<SnpeModelInterpreter>>
    g_snpe_model_interpreter_register(MODEL_TYPE_SNPE);

}  // namespace TNN_NS
