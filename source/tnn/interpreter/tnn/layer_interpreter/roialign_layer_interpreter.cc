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

DECLARE_LAYER_INTERPRETER(RoiAlign, LAYER_ROIALIGN);

Status RoiAlignLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam** param) {
    auto p = CreateLayerParam<RoiAlignLayerParam>(param);
    GET_INT_1_OR_DEFAULT(p->mode, 1);
    GET_INT_3(p->output_height, p->output_width, p->sampling_ratio);
    GET_FLOAT_1(p->spatial_scale);
    return TNN_OK;
}

Status RoiAlignLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status RoiAlignLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto* layer_param = dynamic_cast<RoiAlignLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }
    output_stream << layer_param->mode << " ";
    output_stream << layer_param->output_height << " ";
    output_stream << layer_param->output_width << " ";
    output_stream << layer_param->sampling_ratio << " ";
    output_stream << layer_param->spatial_scale << " ";
    return TNN_OK;
}

Status RoiAlignLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param, LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(RoiAlign, LAYER_ROIALIGN);

}  // namespace TNN_NS
