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

DECLARE_LAYER_INTERPRETER(PRelu, LAYER_PRELU);

Status PReluLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    PReluLayerParam* layer_param = new PReluLayerParam();
    *param                       = layer_param;
    int index                    = start_index;

    if (index < layer_cfg_arr.size()) {
        layer_param->channel_shared = (atoi(layer_cfg_arr[index++].c_str()) == 1) ? 1 : 0;
    }

    if (index < layer_cfg_arr.size()) {
        layer_param->has_filler = (atoi(layer_cfg_arr[index++].c_str()) == 1) ? 1 : 0;
    }
    return TNN_OK;
}

Status PReluLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    PReluLayerResource* layer_res = new PReluLayerResource();
    *resource                     = layer_res;

    layer_res->name = deserializer.GetString();

    RawBuffer k;
    deserializer.GetRaw(k);
    layer_res->slope_handle = k;
    return TNN_OK;
}

Status PReluLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto layer_param = dynamic_cast<PReluLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    output_stream << layer_param->channel_shared << " " << layer_param->has_filler << " ";
    return TNN_OK;
}

Status PReluLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    auto layer_res = dynamic_cast<PReluLayerResource*>(resource);
    if (nullptr == layer_res) {
        LOGE("invalid layer res to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer res to save");
    }
    serializer.PutString(layer_res->name);
    serializer.PutRaw(layer_res->slope_handle);
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(PRelu, LAYER_PRELU);

}  // namespace TNN_NS
