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
#include "base_layer.h"

namespace TNN_NS {

DECLARE_LAYER(DetectionPostProcess, LAYER_DETECTION_POST_PROCESS);

Status DetectionPostProcessLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status DetectionPostProcessLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    assert(input_blobs_.size() == 2);
    assert(output_blobs_.size() == 4);
    auto param                   = dynamic_cast<DetectionPostProcessLayerParam*>(param_);
    CHECK_PARAM_NULL(param);
    const int num_detected_boxes = param->max_detections * param->max_classes_per_detection;
    const int bath_size          = input_blobs_[0]->GetBlobDesc().dims[0];
    // Outputs: detection_boxes, detection_classes, detection_scores, num_detections
    auto detection_boxes_dims = std::vector<int>();
    detection_boxes_dims.push_back(bath_size);
    detection_boxes_dims.push_back(num_detected_boxes);
    detection_boxes_dims.push_back(4);
    detection_boxes_dims.push_back(1);
    output_blobs_[0]->GetBlobDesc().dims = detection_boxes_dims;

    auto detection_classes_dims = std::vector<int>();
    detection_classes_dims.push_back(bath_size);
    detection_classes_dims.push_back(num_detected_boxes);
    detection_classes_dims.push_back(1);
    detection_classes_dims.push_back(1);
    output_blobs_[1]->GetBlobDesc().dims = detection_classes_dims;

    auto detection_scores_dims = std::vector<int>();
    detection_scores_dims.push_back(bath_size);
    detection_scores_dims.push_back(num_detected_boxes);
    detection_scores_dims.push_back(1);
    detection_scores_dims.push_back(1);
    output_blobs_[2]->GetBlobDesc().dims = detection_scores_dims;

    auto num_detections_dims = std::vector<int>({1, 1, 1, 1});
    output_blobs_[3]->GetBlobDesc().dims = num_detections_dims;
    return TNN_OK;
    // detection_boxes
}

REGISTER_LAYER(DetectionPostProcess, LAYER_DETECTION_POST_PROCESS);
}  // namespace TNN_NS
