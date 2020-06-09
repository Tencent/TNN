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

#ifndef TNN_SOURCE_TNN_INTERPRETER_NCNN_LAYER_INTERPRETER_ABSTRACT_LAYER_INTERPRETER_H_
#define TNN_SOURCE_TNN_INTERPRETER_NCNN_LAYER_INTERPRETER_ABSTRACT_LAYER_INTERPRETER_H_

#include <cstdlib>
#include <string>

#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/ncnn/ncnn_model_interpreter.h"
#include "tnn/interpreter/ncnn/ncnn_param_utils.h"
#include "tnn/interpreter/ncnn/serializer.h"
#include "tnn/utils/split_utils.h"

namespace TNN_NS {

namespace ncnn {

    class AbstractLayerInterpreter {
    public:
        // @brief create layer param form layer cfg array
        virtual Status InterpretProto(std::string type_name, str_dict layer_cfg_arr, LayerType &type,
                                      LayerParam **param) = 0;

        virtual Status InterpretResource(Deserializer &deserializer, std::shared_ptr<LayerInfo> info,
                                         LayerResource **resource) = 0;

        virtual Status SaveProto(std::ofstream &output_stream, LayerParam *param) = 0;

        virtual Status SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource) = 0;

        virtual ~AbstractLayerInterpreter(){};
    };

    template <typename T>
    class TypeLayerInterpreterRegister {
    public:
        TypeLayerInterpreterRegister(std::string type_name) {
            NCNNModelInterpreter::RegisterLayerInterpreter(type_name, new T());
        }
    };

#define DECLARE_LAYER_INTERPRETER(interpreter_name)                                                                    \
    class interpreter_name##LayerInterpreter : public AbstractLayerInterpreter {                                       \
    public:                                                                                                            \
        virtual Status InterpretProto(std::string type_name, str_dict layer_cfg_arr, LayerType &type,                  \
                                      LayerParam **param);                                                             \
        virtual Status InterpretResource(Deserializer &deserializer, std::shared_ptr<LayerInfo> info,                  \
                                         LayerResource **Resource);                                                    \
        virtual Status SaveProto(std::ofstream &output_stream, LayerParam *param) {                                    \
            return TNNERR_LAYER_ERR;                                                                                   \
        }                                                                                                              \
        virtual Status SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource) {              \
            return TNNERR_LAYER_ERR;                                                                                   \
        }                                                                                                              \
    }

#define REGISTER_LAYER_INTERPRETER(interpreter_name, type_name)                                                        \
    TypeLayerInterpreterRegister<interpreter_name##LayerInterpreter> g_##type_name##_layer_interpreter(#type_name);

}  // namespace ncnn

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_NCNN_LAYER_INTERPRETER_ABSTRACT_LAYER_INTERPRETER_H_
