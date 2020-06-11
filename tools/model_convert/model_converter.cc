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

Status ModelConvertor::Init(NetworkConfig& net_config, ModelConfig& model_config, InputShapesMap inputs_shape) {
    DefaultModelInterpreter* interpreter =
        dynamic_cast<DefaultModelInterpreter*>(CreateModelInterpreter(model_config.model_type));
    if (!interpreter) {
        return Status(TNNERR_NET_ERR, "interpreter is nil");
    }
    interpreter_ = std::shared_ptr<DefaultModelInterpreter>(interpreter);

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
