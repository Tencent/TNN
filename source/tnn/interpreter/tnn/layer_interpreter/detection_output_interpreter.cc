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

DECLARE_LAYER_INTERPRETER(DetectionOutput, LAYER_DETECTION_OUTPUT);

Status DetectionOutputLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int index, LayerParam **param) {
    auto p = CreateLayerParam<DetectionOutputLayerParam>(param);

    GET_INT_1(p->num_classes);

    int share_location = 0;
    GET_INT_1(share_location);
    p->share_location = share_location == 0 ? false : true;

    GET_INT_1(p->background_label_id);

    int variance_encoded_in_target = 0;
    GET_INT_1(variance_encoded_in_target);
    p->variance_encoded_in_target = variance_encoded_in_target ? true : false;

    GET_INT_2(p->code_type, p->keep_top_k);
    GET_FLOAT_2(p->confidence_threshold, p->nms_param.nms_threshold);
    GET_INT_1(p->nms_param.top_k);
    GET_FLOAT_1(p->eta);

    return TNN_OK;
}

Status DetectionOutputLayerInterpreter::InterpretResource(Deserializer &deserializer, LayerResource **Resource) {
    return TNN_OK;
}

Status DetectionOutputLayerInterpreter::SaveProto(std::ofstream &output_stream, LayerParam *param) {
    CAST_OR_RET_ERROR(layer_param, DetectionOutputLayerParam, "invalid layer param to save", param);

    output_stream << layer_param->num_classes << " ";
    output_stream << layer_param->share_location << " ";
    output_stream << layer_param->background_label_id << " ";
    output_stream << layer_param->variance_encoded_in_target << " ";
    output_stream << layer_param->code_type << " ";
    output_stream << layer_param->keep_top_k << " ";
    output_stream << layer_param->confidence_threshold << " ";
    output_stream << layer_param->nms_param.nms_threshold << " ";
    output_stream << layer_param->nms_param.top_k << " ";
    output_stream << layer_param->eta << " ";

    return TNN_OK;
}

Status DetectionOutputLayerInterpreter::SaveResource(Serializer &serializer, LayerParam *param,
                                                     LayerResource *resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(DetectionOutput, LAYER_DETECTION_OUTPUT);

}  // namespace TNN_NS
