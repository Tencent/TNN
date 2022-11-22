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

#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

#include <stdlib.h>

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(Cumsum, LAYER_CUMSUM);

Status CumsumLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto cumsum_param = CreateLayerParam<CumsumLayerParam>(param);

    GET_INT_1(cumsum_param->axis);
    GET_INT_1(cumsum_param->exclusive);
    GET_INT_1(cumsum_param->reverse);

    return TNN_OK;
}

Status CumsumLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status CumsumLayerInterpreter::SaveProto(std::ostream& output_stream, LayerParam* param) {
    CAST_OR_RET_ERROR(cumsum_param, CumsumLayerParam, "invalid layer param to save", param);

    output_stream << cumsum_param->axis << " ";
    output_stream << int(cumsum_param->exclusive) << " ";
    output_stream << int(cumsum_param->reverse) << " ";

    return TNN_OK;
}

Status CumsumLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Cumsum, LAYER_CUMSUM);

}  // namespace TNN_NS
