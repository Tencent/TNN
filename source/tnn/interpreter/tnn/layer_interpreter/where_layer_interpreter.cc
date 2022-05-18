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

#include <stdlib.h>

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(Where, LAYER_WHERE);

Status WhereLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    return TNN_OK;
}

Status WhereLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto layer_res = CreateLayerRes<WhereLayerResource>(resource);
    RawBuffer x_buf;
    deserializer.GetRaw(x_buf);
    layer_res->x = x_buf;
    
    RawBuffer y_buf;
    deserializer.GetRaw(y_buf);
    layer_res->y = y_buf;
    
    return TNN_OK;
}

Status WhereLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    return TNN_OK;
}

Status WhereLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    CAST_OR_RET_ERROR(layer_res, WhereLayerResource, "invalid layer res to save", resource);
    serializer.PutRaw(layer_res->x);
    serializer.PutRaw(layer_res->y);
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Where, LAYER_WHERE);

}  // namespace TNN_NS
