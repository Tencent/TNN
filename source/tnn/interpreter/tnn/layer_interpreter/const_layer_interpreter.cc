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

DECLARE_LAYER_INTERPRETER(Const, LAYER_CONST);

Status ConstLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto layer_param = new ConstLayerParam();
    *param           = layer_param;
    int dims_size    = 0;
    if (index < layer_cfg_arr.size()) {
        dims_size = atoi(layer_cfg_arr[index++].c_str());
    }
    for (int i = 0; i < dims_size; ++i) {
        layer_param->dims.push_back(atoi(layer_cfg_arr[index++].c_str()));
    }
    return TNN_OK;
}

Status ConstLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    auto layer_res = CreateLayerRes<ConstLayerResource>(resource);
    RawBuffer buffer;
    deserializer.GetRaw(buffer);
    layer_res->weight_handle = buffer;
    return TNN_OK;
}

Status ConstLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto layer_param = dynamic_cast<ConstLayerParam*>(param);
    output_stream << layer_param->dims.size() << " ";
    for (const auto& dim : layer_param->dims) {
        output_stream << dim << " ";
    }
    return TNN_OK;
}

Status ConstLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    auto layer_res = dynamic_cast<ConstLayerResource*>(resource);
    serializer.PutRaw(layer_res->weight_handle);
    return TNN_OK;
}
REGISTER_LAYER_INTERPRETER(Const, LAYER_CONST);
}  // namespace TNN_NS
