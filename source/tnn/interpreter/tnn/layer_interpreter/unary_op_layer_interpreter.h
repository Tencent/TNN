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

#ifndef TNN_SOURCE_TNN_INTERPRETER_TNN_LAYER_INTERPRETER_UNARY_OP_LAYER_INTERPRETER_H_
#define TNN_SOURCE_TNN_INTERPRETER_TNN_LAYER_INTERPRETER_UNARY_OP_LAYER_INTERPRETER_H_

#include <tnn/core/layer_type.h>
#include <tnn/core/status.h>
#include <tnn/interpreter/layer_resource.h>

#include "abstract_layer_interpreter.h"

namespace TNN_NS {

class UnaryOpLayerInterpreter : public AbstractLayerInterpreter {
public:
    virtual Status InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam **param) {
        return TNN_OK;
    }
    virtual Status InterpretResource(Deserializer &deserializer, LayerResource **Resource) {
        return TNN_OK;
    }
    virtual Status SaveProto(std::ofstream &output_stream, LayerParam *param) {
        return TNN_OK;
    }
    virtual Status SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource) {
        return TNN_OK;
    }
    virtual ~UnaryOpLayerInterpreter(){};
};
}  // namespace TNN_NS

#define REGISTER_UNARY_OP_LAYER_INTERPRETER(type_string, layer_type)                                                   \
    TypeLayerInterpreterRegister<UnaryOpLayerInterpreter> g_##layer_type##_layer_interpreter(layer_type);

#endif  // TNN_SOURCE_TNN_INTERPRETER_TNN_LAYER_INTERPRETER_UNARY_OP_LAYER_INTERPRETER_H_
