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
#include "abstract_layer_interpreter.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(Expand, LAYER_EXPAND);

Status ExpandLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam **param) {
    auto p   = CreateLayerParam<ExpandLayerParam>(param);
    int shape_size = 0;
    GET_INT_1_OR_DEFAULT(shape_size, 0);
    GET_INT_N_INTO_VEC(p->shape, shape_size);
    return TNN_OK;
}

Status ExpandLayerInterpreter::InterpretResource(Deserializer &deserializer, LayerResource **Resource) {
    return TNN_OK;
}

Status ExpandLayerInterpreter::SaveProto(std::ofstream &output_stream, LayerParam *param) {
    CAST_OR_RET_ERROR(layer_param, ExpandLayerParam, "invalid expand param to save", param);
    output_stream << layer_param->shape.size() << " ";
    for (const auto &item : layer_param->shape) {
        output_stream << item << " ";
    }
    return TNN_OK;
}

Status ExpandLayerInterpreter::SaveResource(Serializer &serializer, LayerParam *param, LayerResource *resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Expand, LAYER_EXPAND);

}  // namespace TNN_NS
