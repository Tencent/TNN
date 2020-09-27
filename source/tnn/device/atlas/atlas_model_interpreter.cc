// Copyright 2019 Tencent. All Rights Reserved

#include "atlas_model_interpreter.h"
#include <fstream>
#include "atlas_utils.h"
#include "tnn/utils/split_utils.h"

namespace TNN_NS {

AtlasModelInterpreter::AtlasModelInterpreter() {}

AtlasModelInterpreter::~AtlasModelInterpreter() {}

Status AtlasModelInterpreter::Interpret(std::vector<std::string> params) {
    model_config_.om_str  = params[0];
    model_config_.is_path = false;
    if (model_config_.om_str.length() < 1024) {
        std::ifstream om_file(model_config_.om_str);
        if (!om_file) {
            LOGE("Invalied om file path! (param[0] : %s) take as memory content\n", model_config_.om_str.c_str());
            model_config_.is_path = false;
        } else {
            model_config_.is_path = true;
        }
    }

    return TNN_OK;
}

AtlasModelConfig& AtlasModelInterpreter::GetModelConfig() {
    return model_config_;
}

TypeModelInterpreterRegister<TypeModelInterpreterCreator<AtlasModelInterpreter>> g_atlas_model_interpreter_register(
    MODEL_TYPE_ATLAS);

}  // namespace TNN_NS
