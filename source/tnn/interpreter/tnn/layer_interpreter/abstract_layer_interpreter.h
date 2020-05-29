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

#ifndef TNN_SOURCE_TNN_INTERPRETER_TNN_LAYER_INTERPRETER_ABSTRACT_LAYER_INTERPRETER_H_
#define TNN_SOURCE_TNN_INTERPRETER_TNN_LAYER_INTERPRETER_ABSTRACT_LAYER_INTERPRETER_H_

#include <cstdlib>

#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/tnn/layer_interpreter/layer_interpreter_macro.h"
#include "tnn/interpreter/tnn/model_interpreter.h"
#include "tnn/interpreter/tnn/objseri.h"
#include "tnn/utils/split_utils.h"

using namespace TNN_NS;
namespace TNN_NS {

class AbstractLayerInterpreter {
public:
    // @brief create layer param form layer cfg array
    virtual TNN_NS::Status InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam **param) = 0;

    virtual Status InterpretResource(Deserializer &deserializer, LayerResource **resource) = 0;

    virtual Status SaveProto(std::ofstream &output_stream, LayerParam *param) = 0;

    virtual Status SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource) = 0;

    virtual ~AbstractLayerInterpreter(){};
};

template <typename T>
class TypeLayerInterpreterRegister {
public:
    explicit TypeLayerInterpreterRegister(LayerType type) {
        ModelInterpreter::RegisterLayerInterpreter(type, new T());
    }
};

template <typename T>
T *CreateLayerRes(LayerResource **resource) {
    T *layer_res = new T();
    *resource    = layer_res;
    return layer_res;
}

template <typename T>
T *CreateLayerParam(LayerParam **param) {
    T *layer_param = new T();
    *param         = layer_param;
    return layer_param;
}

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_TNN_LAYER_INTERPRETER_ABSTRACT_LAYER_INTERPRETER_H_
