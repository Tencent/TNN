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

#include "object_detector_ssd.h"

#include <algorithm>

namespace TNN_NS {

ObjectDetectorSSDOutput::~ObjectDetectorSSDOutput() {}

ObjectDetectorSSD::~ObjectDetectorSSD() {}

std::shared_ptr<Mat> ObjectDetectorSSD::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat, std::string name) {
    return TNNSDKSample::ResizeToInputShape(input_mat, name);
}

MatConvertParam ObjectDetectorSSD::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {2.0 / 255, 2.0 / 255, 2.0 / 255, 0.0};
    input_convert_param.bias  = {-1, -1, -1, 0.0};
    return input_convert_param;
}

std::shared_ptr<TNNSDKOutput> ObjectDetectorSSD::CreateSDKOutput() {
    return std::make_shared<ObjectDetectorSSDOutput>();
}

Status ObjectDetectorSSD::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;

    auto output = dynamic_cast<ObjectDetectorSSDOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false, Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));

    auto output_mat_scores = output->GetMat("score");
    auto output_mat_boxes  = output->GetMat("output");
    RETURN_VALUE_ON_NEQ(!output_mat_scores, false, Status(TNNERR_PARAM_ERR, "output_mat_scores is invalid"));
    RETURN_VALUE_ON_NEQ(!output_mat_boxes, false, Status(TNNERR_PARAM_ERR, "output_mat_boxes is invalid"));

    auto input_shape = GetInputShape();
    RETURN_VALUE_ON_NEQ(input_shape.size() == 4, true, Status(TNNERR_PARAM_ERR, "GetInputShape is invalid"));

    std::vector<ObjectInfo> object_list;
    GenerateObjects(object_list, output_mat_scores, output_mat_boxes, 0.75, input_shape[3], input_shape[2]);

    std::vector<ObjectInfo> object_list_nms;
    TNN_NS::NMS(object_list, object_list_nms, 0.25, TNNHardNMS);
    output->object_list = object_list_nms;
    return status;
}

void ObjectDetectorSSD::GenerateObjects(std::vector<ObjectInfo> &objects, std::shared_ptr<Mat> scores,
                                        std::shared_ptr<Mat> boxes, float score_threshold, int image_width,
                                        int image_height) {
    int num_anchors = scores->GetDim(1);
    int num_class   = scores->GetDim(2);

    float *scores_data = (float *)scores->GetData();
    float *boxes_data  = (float *)boxes->GetData();

    auto clip = [](float v) { return (std::min)(v > 0.0 ? v : 0.0, 1.0); };

    for (int i = 0; i < num_anchors; i++) {
        int target_class_id      = 0;
        float target_class_score = -1;
        for (int class_id = 0; class_id < num_class; class_id++) {
            int index = i * num_class + class_id;
            if (scores_data[index] > target_class_score) {
                target_class_id    = class_id;
                target_class_score = scores_data[index];
            }
        }

        if (target_class_score <= score_threshold || target_class_id == 0) {
            continue;
        }

        float y_c = boxes_data[i * 4 + 0];
        float x_c = boxes_data[i * 4 + 1];
        float th  = boxes_data[i * 4 + 2];
        float tw  = boxes_data[i * 4 + 3];

        ObjectInfo info;
        info.image_width  = image_width;
        info.image_height = image_height;
        info.class_id     = target_class_id;
        info.score        = target_class_score;

        info.x1 = clip(x_c - tw / 2.0) * image_width;
        info.y1 = clip(y_c - th / 2.0) * image_height;
        info.x2 = clip(x_c + tw / 2.0) * image_width;
        info.y2 = clip(y_c + th / 2.0) * image_height;

        objects.push_back(info);
    }
}

}  // namespace TNN_NS
