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
#include "tnn/core/layer_type.h"
#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"

namespace TNN_NS {

DECLARE_LAYER_INTERPRETER(DetectionPostProcess, LAYER_DETECTION_POST_PROCESS);

Status DetectionPostProcessLayerInterpreter::InterpretProto(str_arr layer_cfg_arr, int start_index,
                                                            LayerParam **param) {
    int index = start_index;
    auto p    = CreateLayerParam<DetectionPostProcessLayerParam>(param);
    GET_INT_1(p->max_detections);
    GET_INT_1(p->max_classes_per_detection);
    GET_INT_1(p->detections_per_class);
    int use_regular_nms = 0;
    GET_INT_1(use_regular_nms);
    p->use_regular_nms = use_regular_nms == 0 ? false : true;
    GET_FLOAT_1(p->nms_score_threshold);
    GET_FLOAT_1(p->nms_iou_threshold);
    GET_INT_1(p->num_classes);
    float y_scale = 1, x_scale = 1, h_scale = 1, w_scale = 1;
    GET_FLOAT_1(y_scale);
    GET_FLOAT_1(x_scale);
    GET_FLOAT_1(h_scale);
    GET_FLOAT_1(w_scale);
    p->center_size_encoding.push_back(y_scale);
    p->center_size_encoding.push_back(x_scale);
    p->center_size_encoding.push_back(h_scale);
    p->center_size_encoding.push_back(w_scale);
    int has_anchors = 0;
    GET_INT_1(has_anchors);
    p->has_anchors = has_anchors == 0 ? false : true;
    GET_INT_1(p->num_anchors);
    GET_INT_1(p->anchors_coord_num);
    return TNN_OK;
}

Status DetectionPostProcessLayerInterpreter::InterpretResource(Deserializer &deserializer, LayerResource **resource) {
    auto layer_res = CreateLayerRes<DetectionPostProcessLayerResource>(resource);

    std::string layer_name = deserializer.GetString();
    GET_BUFFER_FOR_ATTR(layer_res, anchors_handle, deserializer);
    return TNN_OK;
}

Status DetectionPostProcessLayerInterpreter::SaveProto(std::ofstream &output_stream, LayerParam *param) {
    CAST_OR_RET_ERROR(layer_param, DetectionPostProcessLayerParam, "invalid layer param to save", param);

    output_stream << layer_param->max_detections << " ";
    output_stream << layer_param->max_classes_per_detection << " ";
    output_stream << layer_param->detections_per_class << " ";
    output_stream << layer_param->use_regular_nms << " ";
    output_stream << layer_param->nms_score_threshold << " ";
    output_stream << layer_param->nms_iou_threshold << " ";
    output_stream << layer_param->num_classes << " ";
    for (auto scale : layer_param->center_size_encoding) {
        output_stream << scale << " ";
    }
    output_stream << layer_param->has_anchors << " ";
    output_stream << layer_param->num_anchors << " ";
    output_stream << layer_param->anchors_coord_num << " ";
    return TNN_OK;
}

Status DetectionPostProcessLayerInterpreter::SaveResource(Serializer &serializer, LayerParam *param,
                                                          LayerResource *resource) {
    CAST_OR_RET_ERROR(layer_param, DetectionPostProcessLayerParam, "invalid layer param", param);
    CAST_OR_RET_ERROR(layer_res, DetectionPostProcessLayerResource, "invalid layer resource", resource);
    serializer.PutString(layer_res->name);
    serializer.PutRaw(layer_res->anchors_handle);
    return TNN_OK;
}

REGISTER_LAYER_INTERPRETER(DetectionPostProcess, LAYER_DETECTION_POST_PROCESS);

}  // namespace TNN_NS