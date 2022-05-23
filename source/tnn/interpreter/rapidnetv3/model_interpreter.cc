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
        return number == g_version_magic_number_rapidnet_v3 || number == g_version_magic_number_tnn ||
               number == g_version_magic_number_tnn_v2;
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
        } else if (g_version_magic_number_tnn_v2 == this->version_magic_number) {
            return MV_TNN_V2;
        } else {
            return MV_RPNV1;
        }
    }

    Status ModelInterpreter::InterpretProto(std::string &content) {
        return TNN_NS::ModelInterpreter::InterpretProto(content);
    }

    Status ModelInterpreter::InterpretInput(const std::string &inputs_content) {
        if (this->version_magic_number == 0 || this->version_magic_number == g_version_magic_number_rapidnet_v3) {
            // RPN OR RPN_V3
            NetStructure *structure = GetNetStructure();
            str_arr inputs_cfg_vec;
            Status ret = SplitUtils::SplitStr(inputs_content.c_str(), inputs_cfg_vec, ":", true, false);
            if (ret != TNN_OK) {
                return Status(TNNERR_INVALID_NETCFG, "split input line error");
            }
            /*
             * input list is separated by : symbol
             * eg:
             *  input0 1 3 384 128 : input1 1 3 64 64
             */
            for (int i = 0; i < inputs_cfg_vec.size(); i++) {
                str_arr input_cfg_vec;
                ret = SplitUtils::SplitStr(inputs_cfg_vec[i].c_str(), input_cfg_vec, " ", true, false);
                if (ret != TNN_OK || input_cfg_vec.size() < input_layer_cfg_count) {
                    return Status(TNNERR_INVALID_NETCFG, "split input line error");
                }
                DimsVector &input_shape = structure->inputs_shape_map[input_cfg_vec[0]];
                // input_shape.set_name(input_cfg_vec[0]);
                for (int dim_i = 1; dim_i < input_cfg_vec.size(); dim_i++) {
                    input_shape.push_back(atoi(input_cfg_vec[dim_i].c_str()));
                }
            }
            return TNN_OK;
        } else {
            // TNN or TNN_v2
            return TNN_NS::ModelInterpreter::InterpretInput(inputs_content);
        }
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

    const safe_map<LayerType, std::shared_ptr<TNN_NS::AbstractLayerInterpreter>>
        &ModelInterpreter::GetLayerInterpreterMap() {
            return TNN_NS::ModelInterpreter::GetLayerInterpreterMap();
    }

}  // namespace rapidnetv3
