// Copyright 2019 Tencent. All Rights Reserved

#include "tnn/interpreter/rapidnetv3/model_interpreter.h"
#include <sstream>
#include <stdlib.h>
#include "tnn/core/tnn_impl_default.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/interpreter/rapidnetv3/encryption/encryption.h"
#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"
#include "tnn/interpreter/rapidnetv3/objseri.h"



namespace rapidnetv3 {
    TNNImplFactoryRegister<TNNImplFactory<TNNImplDefault>>
g_default_factory_register_rapidnet_v3(MODEL_TYPE_RAPIDNET);

    TypeModelInterpreterRegister<
        TypeModelInterpreterCreator<ModelInterpreter>>
        g_rapidnet_v3_model_interpreter_register(MODEL_TYPE_RAPIDNET);

    std::string ModelInterpreter::Transfer(std::string content) {
        if (this->version_magic_number == g_version_magic_number_rapidnet_v3) {
            content = BlurMix(content.c_str(), (int)content.length(), false);
        }
        return content;
    }

    bool ModelInterpreter::IsValidVersionNumber(uint32_t number) {
        return number == g_version_magic_number_rapidnet_v3 ||
        number == g_version_magic_number_tnn;
    }

    std::shared_ptr<TNN_NS::Deserializer> ModelInterpreter::GetDeserializer(std::istream &is) {
        return std::make_shared<rapidnetv3::Deserializer>(is);
    }

    Status ModelInterpreter::Interpret(std::vector<std::string> &params) {
         return TNN_NS::ModelInterpreter::Interpret(params);
    }

    ModelVersion ModelInterpreter::GetModelVersion() {
        if (g_version_magic_number_rapidnet_v3 == this->version_magic_number) {
            return MV_RPNV3;
        } else if (g_version_magic_number_tnn == this->version_magic_number) {
            return MV_TNN;
        } else {
            return MV_RPNV1;
        }
    }

    Status ModelInterpreter::InterpretProto(std::string &content) {
        return TNN_NS::ModelInterpreter::InterpretProto(content);
    }

    Status ModelInterpreter::InterpretInput(const std::string& inputs_content) {
        return TNN_NS::ModelInterpreter::InterpretInput(inputs_content);
    }

    Status ModelInterpreter::InterpretOutput(const std::string& outputs_content) {
        return TNN_NS::ModelInterpreter::InterpretOutput(outputs_content);
    }

    Status ModelInterpreter::InterpretLayer(const std::string& layer_str) {
       return TNN_NS::ModelInterpreter::InterpretLayer(layer_str);
    }

    Status ModelInterpreter::InterpretModel(std::string &model_content) {
        return TNN_NS::ModelInterpreter::InterpretModel(model_content);
    }

    Status ModelInterpreter::RegisterLayerInterpreter(
        LayerType type, TNN_NS::AbstractLayerInterpreter *interpreter) {
        return TNN_NS::ModelInterpreter::RegisterLayerInterpreter(type, interpreter);
    }

    std::map<LayerType, std::shared_ptr<TNN_NS::AbstractLayerInterpreter>>
        &ModelInterpreter::GetLayerInterpreterMap() {
            return TNN_NS::ModelInterpreter::GetLayerInterpreterMap();
    }

}  // namespace rapidnetv3
