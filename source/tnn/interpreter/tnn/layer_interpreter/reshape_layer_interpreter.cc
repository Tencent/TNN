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

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

#include <stdlib.h>

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(Reshape, LAYER_RESHAPE);

Status ReshapeLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto p = CreateLayerParam<ReshapeLayerParam>(param);
    GET_INT_2(p->axis, p->num_axes);

    int top_blob_dim_size;
    GET_INT_1_OR_DEFAULT(top_blob_dim_size, -1);
    if (top_blob_dim_size == -1) {
        LOGE("Error: ReshapeLayerInterpreter: invalid layer param\n");
        return Status(TNNERR_PARAM_ERR, "ReshapeLayerInterpreter: invalid layer param");
    }

    p->shape.clear();
    GET_INT_N_INTO_VEC(p->shape, top_blob_dim_size);

    GET_INT_1_OR_DEFAULT(p->reshape_type, 0);

    return TNN_OK;
}

Status ReshapeLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status ReshapeLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    CAST_OR_RET_ERROR(layer_param, ReshapeLayerParam, "invalid reshape param to save", param);

    output_stream << layer_param->axis << " ";
    output_stream << layer_param->num_axes << " ";
    output_stream << layer_param->shape.size() << " ";
    for (auto item : layer_param->shape) {
        output_stream << item << " ";
    }
    output_stream << layer_param->reshape_type << " ";

    return TNN_OK;
}

Status ReshapeLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Reshape, LAYER_RESHAPE);

}  // namespace TNN_NS
