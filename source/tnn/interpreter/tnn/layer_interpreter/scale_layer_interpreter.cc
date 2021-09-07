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

DECLARE_LAYER_INTERPRETER(Scale, LAYER_SCALE);

Status ScaleLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    ScaleLayerParam* layer_param = new ScaleLayerParam();
    *param                       = layer_param;
    int index                    = start_index;

    layer_param->axis      = atoi(layer_cfg_arr[index++].c_str());
    layer_param->num_axes  = atoi(layer_cfg_arr[index++].c_str());
    layer_param->bias_term = atoi(layer_cfg_arr[index++].c_str());

    return TNN_OK;
}

Status ScaleLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    BatchNormLayerResource* layer_res = new BatchNormLayerResource();
    *resource                         = layer_res;

    std::string layer_name = deserializer.GetString();

    int has_bias = deserializer.GetInt();

    RawBuffer scale;
    deserializer.GetRaw(scale);

    RawBuffer bias;
    if (has_bias)
        deserializer.GetRaw(bias);

    layer_res->scale_handle = scale;
    layer_res->bias_handle  = bias;

    return TNN_OK;
}

Status ScaleLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    ScaleLayerParam* layer_param = dynamic_cast<ScaleLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }

    output_stream << layer_param->axis << " ";
    output_stream << layer_param->num_axes << " ";
    output_stream << layer_param->bias_term << " ";

    return TNN_OK;
}

Status ScaleLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    ScaleLayerParam* layer_param = dynamic_cast<ScaleLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    BatchNormLayerResource* layer_res = dynamic_cast<BatchNormLayerResource*>(resource);
    if (nullptr == layer_res) {
        LOGE("invalid layer res to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer res to save");
    }

    serializer.PutString(layer_param->name);
    serializer.PutInt(layer_param->bias_term);
    serializer.PutRaw(layer_res->scale_handle);

    if (layer_param->bias_term) {
        serializer.PutRaw(layer_res->bias_handle);
    }

    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Scale, LAYER_SCALE);

}  // namespace TNN_NS
