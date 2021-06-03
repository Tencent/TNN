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

DECLARE_LAYER_INTERPRETER(TopK, LAYER_TOPK);

Status TopKLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    auto* layer_param = new TopKLayerParam();
    *param            = layer_param;
    int index         = start_index;

    GET_INT_1_OR_DEFAULT(layer_param->axis, -1);
    GET_INT_1_OR_DEFAULT(layer_param->largest, 1);
    GET_INT_1_OR_DEFAULT(layer_param->sorted, 1);
    GET_INT_1_OR_DEFAULT(layer_param->k, -1);

    return TNN_OK;
}

Status TopKLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status TopKLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {

    CAST_OR_RET_ERROR(layer_param, TopKLayerParam, "invalid topk param to save", param);
    output_stream << layer_param->axis << " " << layer_param->largest << " " << 
                     layer_param->sorted << " " << layer_param->k << " ";
    return TNN_OK;
}

Status TopKLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(TopK, LAYER_TOPK);

}  // namespace TNN_NS
