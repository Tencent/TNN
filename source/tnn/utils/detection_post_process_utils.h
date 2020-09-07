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

#ifndef TNN_SOURCE_TNN_UTILS_DETECTION_POST_PROCESS_UTILS_H_
#define TNN_SOURCE_TNN_UTILS_DETECTION_POST_PROCESS_UTILS_H_
#include "tnn/core/blob.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"

namespace TNN_NS {

struct CenterSizeEncoding {
    float y = 0;
    float x = 0;
    float h = 0;
    float w = 0;
};

struct BoxCornerEncoding {
    float ymin;
    float xmin;
    float ymax;
    float xmax;
};

void DecodeBoxes(DetectionPostProcessLayerParam* param, DetectionPostProcessLayerResource* resource,
                 Blob* boxes_encoding_blob, const CenterSizeEncoding& scale_values, Blob* decode_boxes_blob);

void NonMaxSuppressionMultiClassFastImpl(DetectionPostProcessLayerParam* param,
                                         DetectionPostProcessLayerResource* resource, Blob* decoded_boxes,
                                         Blob* class_predictions, Blob* detection_boxes, Blob* detection_class,
                                         Blob* detection_scores, Blob* num_detections);

void NonMaxSuppressionSingleClasssImpl(Blob* decoded_boxes, const float* scores, int max_detections,
                                       float iou_threshold, float score_threshold, std::vector<int32_t>* selected);

static inline float IOU(const float* boxes, int i, int j);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_UTILS_DETECTION_POST_PROCESS_UTILS_H_
