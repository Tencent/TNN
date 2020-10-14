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

#include "tnn_sdk_sample.h"
#include "pose_detect_landmark.h"
#include "blazepose_landmark.h"
#include "blazeface_detector.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
Status PoseDetectLandmark::Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks) {
    if (sdks.size() < 2) {
        return Status(TNNERR_INST_ERR, "FaceDetectAligner::Init has invalid sdks, its size < 2");
    }

    predictor_detect_ = sdks[0];
    predictor_landmark_ = sdks[1];
    return TNNSDKComposeSample::Init(sdks);
}

Status PoseDetectLandmark::Predict(std::shared_ptr<TNNSDKInput> sdk_input,
                                  std::shared_ptr<TNNSDKOutput> &sdk_output) {
    Status status = TNN_OK;

    if (!sdk_input || sdk_input->IsEmpty()) {
        status = Status(TNNERR_PARAM_ERR, "input image is empty ,please check!");
        LOGE("input image is empty ,please check!\n");
        return status;
    }
    auto predictor_detect_async = predictor_detect_;
    auto predictor_landmark_async = predictor_landmark_;
    auto predictor_detect_cast = dynamic_cast<BlazePoseDetector *>(predictor_detect_async.get());
    auto predictor_landmark_cast = dynamic_cast<BlazePoseLandmark *>(predictor_landmark_async.get());

    auto image_mat = sdk_input->GetMat();
    origin_input_shape = image_mat->GetDims();
    const int image_orig_height = image_mat->GetHeight();
    const int image_orig_width = image_mat->GetWidth();

    // output of each model
    std::shared_ptr<TNNSDKOutput> sdk_output_face = nullptr;
    std::shared_ptr<TNNSDKOutput> sdk_output_mesh = nullptr;

    // phase1: blazepose detect
    {
    }

    // phase2: blazepose landmark
    {
    }

    //get output
    {
    }
    return TNN_OK;
}

#define M_PI        3.14159265358979323846264338327950288

static inline float NormalizeRadians(float angle) {
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

void PoseDetectLandmark::Detection2ROI(std::vector<BlazePoseInfo>& detects, ROIRect& roi) {
    constexpr int start_kp_idx = 2;
    constexpr int end_kp_idx = 3;
    constexpr float target_angle = 90.0 / 180.0;
    // only use the first detect
    const auto& detect = detects[0];
    const int input_height = origin_input_shape[2];
    const int input_width  = origin_input_shape[3];
    float x_center = detect.key_points[start_kp_idx].first * input_width;
    float y_center = detect.key_points[start_kp_idx].second * input_height;
    float x_scale  = detect.key_points[end_kp_idx].first * input_width;
    float y_scale  = detect.key_points[end_kp_idx].second * input_height;

    // bounding box size as double distance from center to scale point.
    const float box_size = std::sqrt((x_scale - x_center) * (x_scale - x_center) +
                    (y_scale - y_center) * (y_scale - y_center)) * 2.0;
    // rotation
    float rotation = NormalizeRadians(target_angle - std::atan2(-(y_scale - y_center), x_scale - x_center));
    // resulting bounding box.
    x_center = x_center / input_width;
    y_center = y_center / input_height;
    float width  = box_size / input_width;
    float height = box_size / input_height;
    const float long_side = std::max(width * input_width, height * input_height);
    width  = long_side  / input_width;
    height = long_side / input_height;
    // TODO: we need to set these to the blazepose_landmark model
    roi.x_center = x_center;
    roi.y_center = y_center;
    roi.width  = width;
    roi.height = height;
    roi.rotation = rotation;
}

}

