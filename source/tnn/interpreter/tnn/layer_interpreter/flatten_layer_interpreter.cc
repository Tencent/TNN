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

DECLARE_LAYER_INTERPRETER(Flatten, LAYER_FLATTEN);

Status FlattenLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    ReshapeLayerParam* layer_param = new ReshapeLayerParam();
    *param                         = layer_param;
    int index                      = start_index;

    layer_param->axis = atoi(layer_cfg_arr[index++].c_str());

    layer_param->num_axes = atoi(layer_cfg_arr[index++].c_str());
    if (-1 == layer_param->num_axes)
        layer_param->num_axes = 4;

    layer_param->shape.clear();
    if (layer_param->axis == 1) {
        layer_param->shape.resize(3);
        layer_param->shape[0] = 1;
        layer_param->shape[1] = -1;
        layer_param->shape[2] = 1;
    } else if (layer_param->axis == 0) {
        layer_param->shape.resize(4);
        layer_param->shape[0] = 0;
        layer_param->shape[1] = -1;
        layer_param->shape[2] = 1;
        layer_param->shape[3] = 1;
    } else {
        LOGE("flatten param axis unsupported");
        return Status(TNNERR_PARAM_ERR, "flatten param axis unsupported");
    }

    return TNN_OK;
}

Status FlattenLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status FlattenLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    ReshapeLayerParam* layer_param = static_cast<ReshapeLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }

    output_stream << layer_param->axis << " ";
    output_stream << layer_param->num_axes << " ";

    return TNN_OK;
}

Status FlattenLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(Flatten, LAYER_FLATTEN);

}  // namespace TNN_NS
