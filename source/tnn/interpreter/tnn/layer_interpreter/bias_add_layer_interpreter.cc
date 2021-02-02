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

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(BiasAdd, LAYER_BIAS_ADD);

Status BiasAddLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    return TNN_OK;
}

Status BiasAddLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto bias_res = CreateLayerRes<BatchNormLayerResource>(resource);
    GET_BUFFER_FOR_ATTR(bias_res, bias_handle, deserializer);
    return TNN_OK;
}

Status BiasAddLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    return TNN_OK;
}

Status BiasAddLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    CAST_OR_RET_ERROR(bias_res, BiasAddLayerResource, "invalid layer res to save", resource);
    serializer.PutRaw(bias_res->bias_handle);
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(BiasAdd, LAYER_BIAS_ADD);

}  // namespace TNN_NS
