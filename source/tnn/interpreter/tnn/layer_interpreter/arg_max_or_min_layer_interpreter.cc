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

DECLARE_LAYER_INTERPRETER(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

Status ArgMaxOrMinLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto p = CreateLayerParam<ArgMaxOrMinLayerParam>(param);


    if (index < layer_cfg_arr.size()) {
        p->mode = atoi(layer_cfg_arr[index++].c_str());
    }

    if (index < layer_cfg_arr.size()) {
        p->axis = atoi(layer_cfg_arr[index++].c_str());
    }

    if (index < layer_cfg_arr.size()) {
        p->keep_dims = atoi(layer_cfg_arr[index++].c_str());
    }

    if (index < layer_cfg_arr.size()) {
        p->select_last_index = atoi(layer_cfg_arr[index++].c_str());
    }
    return TNN_OK;
}

Status ArgMaxOrMinLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status ArgMaxOrMinLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
	auto layer_param = dynamic_cast<ArgMaxOrMinLayerParam*>(param);
    CHECK_PARAM_NULL(layer_param);
    output_stream << layer_param->mode << " ";
    output_stream << layer_param->axis << " ";
    output_stream << layer_param->keep_dims << " ";
    output_stream << layer_param->select_last_index << " ";

    return TNN_OK;
}

Status ArgMaxOrMinLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

}  // namespace TNN_NS
