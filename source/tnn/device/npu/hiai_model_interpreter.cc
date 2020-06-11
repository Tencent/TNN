// Copyright 2019 Tencent. All Rights Reserved

#include "hiai_model_interpreter.h"
#include <fstream>

namespace TNN_NS {

HiaiModelInterpreter::HiaiModelInterpreter() {}

HiaiModelInterpreter::~HiaiModelInterpreter() {}

Status HiaiModelInterpreter::Interpret(std::vector<std::string> params) {
    std::string model_path = params[1];
    std::ifstream model_file(model_path);
    if (!model_file) {
        LOGE("Invalied model file path!\n");
        return TNNERR_INVALID_MODEL;
    }

    return TNN_OK;
}

TypeModelInterpreterRegister<TypeModelInterpreterCreator<HiaiModelInterpreter>>
    g_hiai_model_interpreter_register(MODEL_TYPE_HIAI);

}  // namespace TNN_NS
