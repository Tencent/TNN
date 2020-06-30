// Copyright 2019 Tencent. All Rights Reserved

#include "atlas_model_interpreter.h"
#include <fstream>
#include "atlas_utils.h"
#include "tnn/utils/split_utils.h"

namespace TNN_NS {

AtlasModelInterpreter::AtlasModelInterpreter() {}

AtlasModelInterpreter::~AtlasModelInterpreter() {}

Status AtlasModelInterpreter::Interpret(std::vector<std::string> params) {
    model_config_.om_path = params[0];
    std::ifstream om_file(model_config_.om_path);
    if (!om_file) {
        LOGE("Invalied om file!\n");
        return TNNERR_INVALID_MODEL;
    }

    return TNN_OK;
}

AtlasModelConfig& AtlasModelInterpreter::GetModelConfig() {
    return model_config_;
}

TypeModelInterpreterRegister<TypeModelInterpreterCreator<AtlasModelInterpreter>> g_atlas_model_interpreter_register(
    MODEL_TYPE_ATLAS);

}  // namespace TNN_NS
