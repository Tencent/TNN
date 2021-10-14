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

#include <stdlib.h>

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(Less, LAYER_LESS);

Status LessLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto p = CreateLayerParam<MultidirBroadcastLayerParam>(param);
    GET_INT_1_OR_DEFAULT(p->weight_input_index, 1);
    return TNN_OK;
}

Status LessLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto layer_res = CreateLayerRes<EltwiseLayerResource>(resource);
    GET_BUFFER_FOR_ATTR(layer_res, element_handle, deserializer);
    return TNN_OK;
}

Status LessLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    CAST_OR_RET_ERROR(layer_param, MultidirBroadcastLayerParam, "invalid layer param to save", param);
    output_stream << layer_param->weight_input_index << " ";
    return TNN_OK;
}

Status LessLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    CAST_OR_RET_ERROR(layer_res, EltwiseLayerResource, "invalid layer res to save", resource);
    serializer.PutRaw(layer_res->element_handle);
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Less, LAYER_LESS);

}  // namespace TNN_NS
