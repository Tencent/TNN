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

    DECLARE_LAYER_INTERPRETER(BinaryOp);

    REGISTER_LAYER_INTERPRETER(BinaryOp, BinaryOp);

    static std::map<int, LayerType> global_layer_type_map = {
        {0, LAYER_ADD},         {1, LAYER_SUB},     {2, LAYER_MUL},   {3, LAYER_DIV},
        {4, LAYER_MAXIMUM},     {5, LAYER_MINIMUM}, {6, LAYER_POWER}, {7, LAYER_NOT_SUPPORT},  // RSUB
        {8, LAYER_NOT_SUPPORT},                                                                // RDIV
    };

    Status BinaryOpLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                    LayerParam** param) {
        MultidirBroadcastLayerParam* layer_param = new MultidirBroadcastLayerParam();
        *param                                   = layer_param;

        auto& p = param_dict;

        int op_type     = GetInt(p, 0, 0);
        int with_scalar = GetInt(p, 1, 0);
        float b         = GetFloat(p, 2, 0.f);

        type = global_layer_type_map[op_type];

        if (with_scalar != 0) {
            LOGET("BinaryOp with scaler not supported\n", "ncnn");
            type = LAYER_NOT_SUPPORT;
        }

        return TNN_OK;
    }

    Status BinaryOpLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                       LayerResource** resource) {
        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
