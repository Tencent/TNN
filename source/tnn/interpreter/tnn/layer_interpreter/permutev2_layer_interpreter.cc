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

DECLARE_LAYER_INTERPRETER(PermuteV2, LAYER_PERMUTEV2);

Status PermuteV2LayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    auto layer_param = new PermuteV2LayerParam();
    *param           = layer_param;
    int index        = start_index;

    layer_param->orders.clear();
    layer_param->dim0 = atoi(layer_cfg_arr[index++].c_str());
    layer_param->dim1 = atoi(layer_cfg_arr[index++].c_str());

    return TNN_OK;
}

Status PermuteV2LayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status PermuteV2LayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    PermuteV2LayerParam* layer_param = dynamic_cast<PermuteV2LayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }

    output_stream << layer_param->dim0 << " ";
    output_stream << layer_param->dim1 << " ";

    return TNN_OK;
}

Status PermuteV2LayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(PermuteV2, LAYER_PERMUTEV2);

}  // namespace TNN_NS
