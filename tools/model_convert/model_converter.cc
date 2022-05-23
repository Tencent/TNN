// Copyright 2019 Tencent. All Rights Reserved

#include "model_converter.h"
#include <algorithm>
#include <cmath>
#include <random>
#include "tnn/core/macro.h"
#include "tnn/interpreter/rapidnetv3/model_packer.h"

namespace TNN_NS {

ModelConvertor::ModelConvertor() {
    model_version_ = rapidnetv3::MV_RPNV3;
}

ModelConvertor::~ModelConvertor() {}

Status ModelConvertor::Init(ModelConfig& model_config) {
    rapidnetv3::ModelInterpreter* interpreter =
        dynamic_cast<rapidnetv3::ModelInterpreter*>(CreateModelInterpreter(MODEL_TYPE_RAPIDNET));
    if (!interpreter) {
        return Status(TNNERR_NET_ERR, "interpreter is nil");
    }
    interpreter_ = std::shared_ptr<rapidnetv3::ModelInterpreter>(interpreter);

    Status status = interpreter_->Interpret(model_config.params);
    if (status != TNN_OK) {
        LOGE("interpret the model falied! (%s)\n", status.description().c_str());
        return TNNERR_INVALID_MODEL;
    }

    return TNN_OK;
}

void ModelConvertor::SetModelVersion(rapidnetv3::ModelVersion ver) {
    model_version_ = ver;
}

void ModelConvertor::DumpModelInfo() {
    ModelInfo model_info;
    model_info.model_version   = interpreter_->GetModelVersion();
    model_info.model_data_type = GetNetResourceDataType(interpreter_->GetNetResource());

    printf("model info: \n");
    // dump model_version
    printf("\tmodel version: ");
    if (rapidnetv3::MV_RPNV1 == model_info.model_version) {
        printf("Rapidnet V1\n");
    } else if (rapidnetv3::MV_TNN == model_info.model_version) {
        printf("TNN\n");
    } else if (rapidnetv3::MV_RPNV3 == model_info.model_version) {
        printf("Rapidnet V3\n");
    } else {
        printf("unknown\n");
    }

    // dump model data type
    printf("\tmodel data type: ");
    if (DATA_TYPE_FLOAT == model_info.model_data_type) {
        printf("FP32\n");
    } else if (DATA_TYPE_HALF == model_info.model_data_type) {
        printf("FP16\n");
    } else if (DATA_TYPE_INT8 == model_info.model_data_type) {
        printf("INT8\n");
    } else {
        printf("unknown\n");
    }
}

Status ModelConvertor::Serialize(std::string proto_path, std::string model_path) {
    NetStructure* net_struct  = interpreter_->GetNetStructure();
    NetResource* net_resource = interpreter_->GetNetResource();
    if (net_struct == nullptr || net_resource == nullptr) {
        LOGE("net struct or net resource is null\n");
        return TNNERR_INVALID_MODEL;
    }

    rapidnetv3::ModelPacker packer(net_struct, net_resource);
    packer.SetVersion(model_version_);

    Status status = packer.Pack(proto_path, model_path);
    if (status != TNN_OK) {
        LOGE("pack the model falied!\n");
    }

    return TNN_OK;
}

}  // namespace TNN_NS
