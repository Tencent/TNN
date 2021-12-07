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

DECLARE_LAYER_INTERPRETER(Div, LAYER_DIV);

Status DivLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    auto layer_param = new MultidirBroadcastLayerParam();
    *param           = layer_param;
    int index        = start_index;

    layer_param->weight_input_index = 1;
    if (index < layer_cfg_arr.size()) {
        layer_param->weight_input_index = atoi(layer_cfg_arr[index++].c_str());
    }

    return TNN_OK;
}

Status DivLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto layer_res = new EltwiseLayerResource();
    *resource      = layer_res;

    // std::string layer_name = deserializer.GetString();

    RawBuffer k;
    deserializer.GetRaw(k);
    layer_res->element_handle = k;

    return TNN_OK;
}

Status DivLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam*>(param);
    output_stream << layer_param->weight_input_index << " ";
    return TNN_OK;
}

Status DivLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    EltwiseLayerResource* layer_res = dynamic_cast<EltwiseLayerResource*>(resource);
    if (nullptr == layer_res) {
        LOGE("invalid layer res to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer res to save");
    }
    serializer.PutRaw(layer_res->element_handle);

    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Div, LAYER_DIV);

}  // namespace TNN_NS
