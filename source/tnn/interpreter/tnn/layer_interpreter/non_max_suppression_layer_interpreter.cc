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

DECLARE_LAYER_INTERPRETER(NonMaxSuppression, LAYER_NON_MAX_SUPPRESSION);

Status NonMaxSuppressionLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index, LayerParam** param) {
    auto* layer_param = new NonMaxSuppressionLayerParam();
    *param            = layer_param;
    int index         = start_index;

    GET_INT_1_OR_DEFAULT(layer_param->center_point_box, 0);
    GET_LONG_1_OR_DEFAULT(layer_param->max_output_boxes_per_class, 0);
    GET_FLOAT_1_OR_DEFAULT(layer_param->iou_threshold, 0.0f);
    GET_FLOAT_1_OR_DEFAULT(layer_param->score_threshold, 0.0f);

    return TNN_OK;
}

Status NonMaxSuppressionLayerInterpreter::InterpretResource(Deserializer& deserializer, LayerResource** resource) {
    return TNN_OK;
}

Status NonMaxSuppressionLayerInterpreter::SaveProto(std::ofstream& output_stream, LayerParam* param) {
    auto* layer_param = static_cast<NonMaxSuppressionLayerParam*>(param);
    if (nullptr == layer_param) {
        LOGE("invalid layer param to save\n");
        return Status(TNNERR_NULL_PARAM, "invalid layer param to save");
    }

    output_stream << layer_param->center_point_box << " " << layer_param->max_output_boxes_per_class << " "
                  << layer_param->iou_threshold << " " << layer_param->score_threshold << " ";

    return TNN_OK;
}

Status NonMaxSuppressionLayerInterpreter::SaveResource(Serializer& serializer, LayerParam* param,
                                                       LayerResource* resource) {
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(NonMaxSuppression, LAYER_NON_MAX_SUPPRESSION);

}  // namespace TNN_NS
