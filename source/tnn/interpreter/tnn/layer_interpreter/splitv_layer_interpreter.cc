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

DECLARE_LAYER_INTERPRETER(SplitV, LAYER_SPLITV);

Status SplitVLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto p = CreateLayerParam<SplitVLayerParam>(param);

    int slice_count = 0;
    GET_INT_2(p->axis, slice_count);

    p->slices.clear();
    GET_INT_N_INTO_VEC(p->slices, slice_count);

    return TNN_OK;
}

Status SplitVLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status SplitVLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    CAST_OR_RET_ERROR(splitv_param, SplitVLayerParam, "invalid layer param to save", param);

    output_stream << splitv_param->axis << " ";
    output_stream << splitv_param->slices.size() << " ";
    for (auto item : splitv_param->slices) {
        output_stream << item << " ";
    }

    return TNN_OK;
}

Status SplitVLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(SplitV, LAYER_SPLITV);

}  // namespace TNN_NS
