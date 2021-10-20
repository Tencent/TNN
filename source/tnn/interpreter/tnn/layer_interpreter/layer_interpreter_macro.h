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

#ifndef TNN_SOURCE_TNN_INTERPRETER_TNN_LAYER_INTERPRETER_LAYER_INTERPRETER_MACRO_H_
#define TNN_SOURCE_TNN_INTERPRETER_TNN_LAYER_INTERPRETER_LAYER_INTERPRETER_MACRO_H_

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

#define DECLARE_LAYER_INTERPRETER(type_string, layer_type)                                                             \
    class type_string##LayerInterpreter : public AbstractLayerInterpreter {                                            \
    public:                                                                                                            \
        virtual Status InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam **param);                     \
        virtual Status InterpretResource(Deserializer &deserializer, LayerResource **resource);                        \
        virtual Status SaveProto(std::ofstream &output_stream, LayerParam *param);                                     \
        virtual Status SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource);               \
    }

#define REGISTER_LAYER_INTERPRETER(type_string, layer_type)                                                            \
    TypeLayerInterpreterRegister<type_string##LayerInterpreter> g_##layer_type##_layer_interpreter(layer_type);

#define DEFAULT_ARR_VAR layer_cfg_arr
#define DEFAULT_INDEX_VAR index

#define GET_LONG_1_OR_DEFAULT(var, default_value)                                                                      \
    do {                                                                                                               \
        if (DEFAULT_INDEX_VAR < DEFAULT_ARR_VAR.size()) {                                                              \
            var = atol(DEFAULT_ARR_VAR[DEFAULT_INDEX_VAR++].c_str());                                                  \
        } else {                                                                                                       \
            var = default_value;                                                                                       \
        }                                                                                                              \
    } while (0)

#define GET_INT_1_OR_DEFAULT(var, default_value)                                                                       \
    do {                                                                                                               \
        if (DEFAULT_INDEX_VAR < DEFAULT_ARR_VAR.size()) {                                                              \
            var = atoi(DEFAULT_ARR_VAR[DEFAULT_INDEX_VAR++].c_str());                                                  \
        } else {                                                                                                       \
            var = default_value;                                                                                       \
        }                                                                                                              \
    } while (0)

#define GET_FLOAT_1_OR_DEFAULT(var, default_value)                                                                     \
    do {                                                                                                               \
        if (DEFAULT_INDEX_VAR < DEFAULT_ARR_VAR.size()) {                                                              \
            var = atof(DEFAULT_ARR_VAR[DEFAULT_INDEX_VAR++].c_str());                                                  \
        } else {                                                                                                       \
            var = default_value;                                                                                       \
        }                                                                                                              \
    } while (0)

#define GET_INT_1(var) GET_INT_1_OR_DEFAULT(var, 0)
#define GET_FLOAT_1(var) GET_FLOAT_1_OR_DEFAULT(var, 0.0f)

#define GET_INT_2(var1, var2)                                                                                          \
    GET_INT_1(var1);                                                                                                   \
    GET_INT_1(var2)
#define GET_FLOAT_2(var1, var2)                                                                                        \
    GET_FLOAT_1(var1);                                                                                                 \
    GET_FLOAT_1(var2)

#define GET_INT_3(var1, var2, var3)                                                                                    \
    GET_INT_2(var1, var2);                                                                                             \
    GET_INT_1(var3)

#define GET_INT_N_INTO_VEC_DEFAULT(vec, n, default_value)                                                              \
    do {                                                                                                               \
        for (int _ii = 0; _ii < n; _ii++) {                                                                            \
            int var = default_value;                                                                                   \
            GET_INT_1_OR_DEFAULT(var, default_value);                                                                  \
            vec.push_back(var);                                                                                        \
        }                                                                                                              \
    } while (0)

#define GET_INT_N_INTO_VEC_REVERSE_DEFAULT(vec, n, default_value)                                                      \
    do {                                                                                                               \
        vec.resize(n);                                                                                                 \
        for (int _ii = n - 1; _ii >= 0; _ii--) {                                                                       \
            int var = default_value;                                                                                   \
            GET_INT_1_OR_DEFAULT(var, default_value);                                                                  \
            vec[_ii] = var;                                                                                            \
        }                                                                                                              \
    } while (0)

#define GET_INT_N_INTO_VEC(vec, n) GET_INT_N_INTO_VEC_DEFAULT(vec, n, 0)

#define GET_INT_N_INTO_VEC_REVERSE(vec, n) GET_INT_N_INTO_VEC_REVERSE_DEFAULT(vec, n, 0)

#define GET_INT_2_INTO_VEC_DEFAULT(vec, default_value) GET_INT_N_INTO_VEC_DEFAULT(vec, 2, default_value)

#define GET_INT_2_INTO_VEC(vec) GET_INT_2_INTO_VEC_DEFAULT(vec, 0)

#define GET_INT_2_INTO_VEC_REVERSE_DEFAULT(vec, default_value) GET_INT_N_INTO_VEC_REVERSE_DEFAULT(vec, 2, default_value)

#define GET_INT_2_INTO_VEC_REVERSE(vec) GET_INT_2_INTO_VEC_REVERSE_DEFAULT(vec, 0)

#define GET_BUFFER_FOR_ATTR(layer_res, attr, deserializer)                                                             \
    do {                                                                                                               \
        RawBuffer buf;                                                                                                 \
        deserializer.GetRaw(buf);                                                                                      \
        layer_res->attr = buf;                                                                                         \
    } while (0)

#define CAST_OR_RET_ERROR(layer_res, T, msg, resource)                                                                 \
    T *layer_res = dynamic_cast<T *>(resource);                                                                        \
    do {                                                                                                               \
        if (nullptr == layer_res) {                                                                                    \
            LOGE(msg);                                                                                                 \
            return Status(TNNERR_NULL_PARAM, msg);                                                                     \
        }                                                                                                              \
    } while (0)

#endif  // TNN_SOURCE_TNN_INTERPRETER_TNN_LAYER_INTERPRETER_LAYER_INTERPRETER_MACRO_H_
