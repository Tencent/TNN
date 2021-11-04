// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tnn/core/tnn_impl_default.h"
#include <fstream>

#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/utils/blob_dump_utils.h"
#include "tnn/interpreter/tnn/model_packer.h"

namespace TNN_NS {

TNNImplFactoryRegister<TNNImplFactory<TNNImplDefault>> g_tnn_impl_default_factory_register(MODEL_TYPE_TNN);

TNNImplFactoryRegister<TNNImplFactory<TNNImplDefault>> g_tnn_impl_ncnn_factory_register(MODEL_TYPE_NCNN);

TNNImplDefault::TNNImplDefault() {}

TNNImplDefault::~TNNImplDefault() {}

Status TNNImplDefault::Init(ModelConfig& config) {
    auto status = TNNImpl::Init(config);
    if (status != TNN_OK) {
        return status;
    }

    auto interpreter = CreateModelInterpreter(config.model_type);
    if (!interpreter) {
        return Status(TNNERR_NET_ERR, "interpreter is nil");
    }
    interpreter_ = std::shared_ptr<AbstractModelInterpreter>(interpreter);
    return interpreter_->Interpret(config.params);
}

Status TNNImplDefault::DeInit() {
    interpreter_ = nullptr;
    return TNN_OK;
}

Status TNNImplDefault::AddOutput(const std::string& layer_name, int output_index) {
    if (!interpreter_) {
        return Status(TNNERR_NET_ERR, "interpreter is nil");
    }

    auto default_interpreter = dynamic_cast<DefaultModelInterpreter*>(interpreter_.get());
    CHECK_PARAM_NULL(default_interpreter);

    default_interpreter->GetNetStructure()->outputs.insert(layer_name);
    return TNN_OK;
}

Status TNNImplDefault::SaveModel(const std::string& proto_path, const std::string& model_path) {
    if(model_config_.model_type != MODEL_TYPE_TNN )
        return Status(TNNERR_NET_ERR, "save trained model only support tnn model for now");
    if(model_config_.params.size() < 0 )
        return Status(TNNERR_NET_ERR, "model config params is empty");
    std::ofstream write_stream;
    write_stream.open(proto_path, std::ios::binary);
    if (!write_stream || !write_stream.is_open() || !write_stream.good()) {
        write_stream.close();
        return Status(TNNERR_PACK_MODEL, "proto file cannot be written");
    }
    write_stream << model_config_.params[0];
    if(!write_stream.good())
        return Status(TNN_TRAIN_ERROR, "write proto error");
    write_stream.close();
    DefaultModelInterpreter* default_interpreter = dynamic_cast<DefaultModelInterpreter*>(interpreter_.get());
    if(!default_interpreter)
        return Status(TNNERR_NET_ERR, "only support default interpreter or the interpreter is null");
    ModelPacker packer(default_interpreter->GetNetStructure(), default_interpreter->GetNetResource());
    return packer.PackModel(model_path);

}

Status TNNImplDefault::GetModelInputShapesMap(InputShapesMap& shapes_map) {
    if (!interpreter_) {
        return Status(TNNERR_NET_ERR, "interpreter is nil");
    }

    auto default_interpreter = dynamic_cast<DefaultModelInterpreter*>(interpreter_.get());
    CHECK_PARAM_NULL(default_interpreter);
    shapes_map = default_interpreter->GetNetStructure()->inputs_shape_map;
    return TNN_OK;
} 


std::shared_ptr<Instance> TNNImplDefault::CreateInst(NetworkConfig& net_config, Status& status,
                                                     InputShapesMap inputs_shape) {
    if (!interpreter_) {
        status = Status(TNNERR_NET_ERR, "interpreter is nil");
        return nullptr;
    }
#if (DUMP_INPUT_BLOB || DUMP_OUTPUT_BLOB)
    //todo: refactor later
    if(net_config.device_type == DEVICE_CUDA) {
        status = AddAllLayersOutput();
        if(status != TNN_OK) {
            return nullptr;
        }
    }
#endif

    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, inputs_shape);

    if (status != TNN_OK) {
        return nullptr;
    }
    return instance;
}

std::shared_ptr<Instance> TNNImplDefault::CreateInst(NetworkConfig& net_config, Status& status,
                                                     InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape) {
    if (!interpreter_) {
        status = Status(TNNERR_NET_ERR, "interpreter is nil");
        return nullptr;
    }
#if (DUMP_INPUT_BLOB || DUMP_OUTPUT_BLOB)
    //todo: refactor later
    if(net_config.device_type == DEVICE_CUDA) {
        status = AddAllLayersOutput();
        if(status != TNN_OK) {
            return nullptr;
        }
    }
#endif

    auto instance = std::make_shared<Instance>(net_config, model_config_);
    status        = instance->Init(interpreter_, min_inputs_shape, max_inputs_shape);

    if (status != TNN_OK) {
        return nullptr;
    }
    return instance;
}

Status TNNImplDefault::AddAllLayersOutput() {
    auto default_interpreter = dynamic_cast<DefaultModelInterpreter*>(interpreter_.get());
    CHECK_PARAM_NULL(default_interpreter);
    auto net_structure = default_interpreter->GetNetStructure();
    for(auto layer_info : net_structure->layers) {
        for(auto output_name : layer_info->outputs) {
            AddOutput(output_name);
        }
    }
    return TNN_OK;
}

}  // namespace TNN_NS
