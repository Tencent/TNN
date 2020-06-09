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

#include "tnn/interpreter/ncnn/layer_interpreter/abstract_layer_interpreter.h"
#include "tnn/interpreter/ncnn/ncnn_layer_type.h"
#include "tnn/interpreter/ncnn/ncnn_param_utils.h"

namespace TNN_NS {

namespace ncnn {

    DECLARE_LAYER_INTERPRETER(UnaryOp);

    REGISTER_LAYER_INTERPRETER(UnaryOp, UnaryOp);

    static std::map<int, LayerType> unary_op_layer_type_map = {
        {0, LAYER_ABS},     {1, LAYER_NEG},     {2, LAYER_FLOOR},   {3, LAYER_CEIL},
        {4, LAYER_SQUARE},  {5, LAYER_SQRT},    {6, LAYER_RSQRT},   {7, LAYER_EXP},
        {8, LAYER_LOG},     {9, LAYER_SIN},     {10,LAYER_COS},     {11,LAYER_TAN},
        {12,LAYER_ASIN},    {13,LAYER_ACOS},    {14,LAYER_ATAN},    {15,LAYER_RECIPROCAL},
        {16,LAYER_TANH},
    };

    Status UnaryOpLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                   LayerParam** param) {
        auto& p = param_dict;
        int op_type = GetInt(p, 0, 0);

        type = unary_op_layer_type_map[op_type];
        
        return TNN_OK;
    }

    Status UnaryOpLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                      LayerResource** resource) {
        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS