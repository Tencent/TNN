// Copyright 2019 Tencent. All Rights Reserved

#include "atlas_model_interpreter.h"
#include <fstream>
#include "atlas_utils.h"
#include "tnn/utils/split_utils.h"

namespace TNN_NS {

static int g_model_count = 0;

AtlasModelInterpreter::AtlasModelInterpreter() {}

AtlasModelInterpreter::~AtlasModelInterpreter() {}

Status AtlasModelInterpreter::Interpret(std::vector<std::string> params) {
    Status ret = ParseConfig(params[0]);
    if (ret != TNN_OK)
        return ret;

    return TNN_OK;
}

AtlasModelConfig& AtlasModelInterpreter::GetModelConfig() {
    return model_config_;
}

Status AtlasModelInterpreter::ParseConfig(std::string config_str) {
    Status ret = TNN_OK;

    std::vector<std::string> str_vec;
    ret = SplitUtils::SplitStr(config_str.c_str(), str_vec, " ", true, false);
    if (ret != TNN_OK) {
        LOGE("invalid config string (format error)!\n");
        return TNNERR_PARAM_ERR;
    }

    if (str_vec.size() != 7) {
        LOGE("invalid config string (item size not match)!\n");
        return TNNERR_PARAM_ERR;
    }

    model_config_.om_path       = str_vec[0];
    model_config_.with_dvpp     = atoi(str_vec[1].c_str()) == 0 ? false : true;
    model_config_.dynamic_aipp  = atoi(str_vec[2].c_str()) == 0 ? false : true;
    model_config_.daipp_swap_rb = atoi(str_vec[3].c_str()) == 0 ? false : true;
    model_config_.daipp_norm    = atoi(str_vec[4].c_str()) == 0 ? false : true;
    model_config_.height        = atoi(str_vec[5].c_str());
    model_config_.width         = atoi(str_vec[6].c_str());

    std::ifstream om_file(model_config_.om_path);
    if (!om_file) {
        LOGE("Invalied om file!\n");
        return TNNERR_INVALID_MODEL;
    }

    model_config_.graph_id = 1000 + g_model_count;
    g_model_count++;

    return TNN_OK;
}

TypeModelInterpreterRegister<TypeModelInterpreterCreator<AtlasModelInterpreter>>
    g_atlas_model_interpreter_register(MODEL_TYPE_ATLAS);

}  // namespace TNN_NS
