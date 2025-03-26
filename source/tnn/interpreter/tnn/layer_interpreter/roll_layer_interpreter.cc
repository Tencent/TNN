// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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
#include "abstract_layer_interpreter.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(Roll, LAYER_ROLL);

Status RollLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam **param) {
    auto p = CreateLayerParam<RollLayerParam>(param);
    int shape_size = 0;
    GET_INT_1_OR_DEFAULT(shape_size, 0);
    GET_INT_N_INTO_VEC(p->shifts, shape_size);
    GET_INT_N_INTO_VEC(p->dims, shape_size);
    return TNN_OK;
}

Status RollLayerInterpreter::InterpretResource(Deserializer &deserializer, LayerResource **Resource) {
    return TNN_OK;
}

Status RollLayerInterpreter::SaveProto(std::ostream &output_stream, LayerParam *param) {
    CAST_OR_RET_ERROR(layer_param, RollLayerParam, "invalid Roll param to save", param);
    output_stream << layer_param->shifts.size() << " ";
    for (const auto &item : layer_param->shifts) {
        output_stream << item << " ";
    }
    for (const auto &item : layer_param->dims) {
        output_stream << item << " ";
    }
    return TNN_OK;
}

Status RollLayerInterpreter::SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Roll, LAYER_ROLL);

}  // namespace TNN_NS
