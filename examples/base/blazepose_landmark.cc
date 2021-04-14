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

#include "blazepose_landmark.h"
#include "time_stamp.h"
#include <cmath>
#include <fstream>
#include <cstring>
#include <algorithm>

namespace TNN_NS {

template <typename KeyPointType, int d>
struct GetCoord {
    static float get(KeyPointType& p) {
        return -1;
    }
};

using KeyPoint2D = std::pair<float, float>;
template<>
struct GetCoord<KeyPoint2D, 0> {
    static float get(KeyPoint2D& p) {
        return p.first;
    }
};

template<>
struct GetCoord<KeyPoint2D, 1> {
    static float get(KeyPoint2D& p) {
        return p.second;
    }
};

using KeyPoint3D = std::tuple<float, float, float>;
template<int d>
struct GetCoord<KeyPoint3D, d> {
    static float get(KeyPoint3D& p) {
        return std::get<d>(p);
    }
};

template <typename KeyPointType>
void BlazePoseLandmark::KeyPoints2RoI(std::vector<KeyPointType>& kps, const RoIGenOptions& option) {
    constexpr double PI = 3.141;
    const int start_kp_idx = option.keypoints_start_idx;
    const int end_kp_idx = option.keypoints_end_idx;
    const float target_angle = option.rotation_target_angle / 180.0 * PI;

    const int input_height = origin_input_shape[2];
    const int input_width  = origin_input_shape[3];

    float x_center = GetCoord<KeyPointType, 0>::get(kps[start_kp_idx]) * input_width;
    float y_center = GetCoord<KeyPointType, 1>::get(kps[start_kp_idx]) * input_height;
    float x_scale  = GetCoord<KeyPointType, 0>::get(kps[end_kp_idx]) * input_width;
    float y_scale  = GetCoord<KeyPointType, 1>::get(kps[end_kp_idx]) * input_height;

    // bounding box size as double distance from center to scale point.
    const float box_size = std::sqrt((x_scale - x_center) * (x_scale - x_center) +
                    (y_scale - y_center) * (y_scale - y_center)) * 2.0;
    // rotation
    float angle = target_angle - std::atan2(-(y_scale - y_center), x_scale - x_center);
    float rotation = angle - 2 * PI * std::floor((angle - (-PI)) / (2 * PI));

    // resulting bounding box.
    x_center = x_center / input_width;
    y_center = y_center / input_height;
    float width  = box_size / input_width;
    float height = box_size / input_height;

    roi.x_center = x_center;
    roi.y_center = y_center;
    roi.width  = width  * option.scale_x;
    roi.height = height * option.scale_y;
    roi.rotation = rotation;
}

void BlazePoseLandmark::GetCropMatrix(float trans_mat[2][3], std::vector<float>&target_size) {
    const int src_height = origin_input_shape[2];
    const int src_width  = origin_input_shape[3];

    float roi_width  = std::round(roi.width  * src_width);
    float roi_height = std::round(roi.height * src_height);
    float roi_x_center = std::round(roi.x_center * src_width);
    float roi_y_center = std::round(roi.y_center * src_height);
    float rotate = roi.rotation;

    // compute resize scale
    auto input_shape = GetInputShape();
    const int input_width  = input_shape[3];
    const int input_height = input_shape[2];
    float scale = std::min({1.0f, input_height/roi.height, input_width/roi.width});
    float target_width  = scale * roi_width;
    float target_height = scale * roi_height;
    trans_mat[0][0] = scale * std::cos(rotate);
    trans_mat[0][1] = scale * std::sin(rotate);
    trans_mat[1][0] = -scale* std::sin(rotate);
    trans_mat[1][1] = scale * std::cos(rotate);

    // compute offset
    float trans_center_x = roi_x_center * trans_mat[0][0] + roi_y_center * trans_mat[0][1];
    float trans_center_y = roi_x_center * trans_mat[1][0] + roi_y_center * trans_mat[1][1];
    const float offset_x = (target_width / 2) - trans_center_x;
    const float offset_y = (target_height/ 2) - trans_center_y;
    trans_mat[0][2] = offset_x;
    trans_mat[1][2] = offset_y;

    target_size.resize(2);
    target_size[0] = target_height;
    target_size[1] = target_width;
}

Status BlazePoseLandmark::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<BlazePoseLandmarkOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));

    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);

    landmark_filter = std::make_shared<VelocityFilter>(this->window_size,
                                                       this->velocity_scale,
                                                       this->min_allowed_object_scale,
                                                       option->fps);

    auto input_dims = GetInputShape();
    option->input_height = input_dims[2];
    option->input_width  = input_dims[3];
    roi_from_prev_frame = false;

    if (option->full_body) {
        // keypoints used to compute roi
        this->roi_option.keypoints_start_idx = 33;
        this->roi_option.keypoints_end_idx = 34;
        // number of total landmarks
        this->num_landmarks = 39;
        // number of visible landmarks
        this->num_visible_landmarks = 33;
        // add lines only for the full-body landmark model
        this->lines.insert(this->lines.end(),
                           this->extended_lines_fb.begin(),
                           this->extended_lines_fb.end());
    }
    return status;
}

std::shared_ptr<Mat> BlazePoseLandmark::ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                                   std::string name) {
    // save the origianl input shape
    origin_input_shape = mat->GetDims();

    auto target_dims   = GetInputShape(name);
    auto target_height = target_dims[2];
    auto target_width  = target_dims[3];

    // step1: crop the roi
    float transMat[2][3];
    std::vector<float> cropSize;
    GetCropMatrix(transMat, cropSize);
    auto crop_shape = target_dims;
    crop_shape[2] = cropSize[0];
    crop_shape[3] = cropSize[1];
    auto cropped_mat = std::make_shared<Mat>(mat->GetDeviceType(), mat->GetMatType(), crop_shape);

    auto status = WarpAffine(mat, cropped_mat, TNNInterpLinear, TNNBorderConstant, transMat);
    RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);

    // step2: transform the cropped
    auto input_height  = cropped_mat->GetHeight();
    auto input_width   = cropped_mat->GetWidth();

    letterbox_pads.fill(0.f);
    const float input_aspect_ratio  = static_cast<float>(input_width) / input_height;
    const float output_aspect_ratio = static_cast<float>(target_width) / target_height;
    if (input_aspect_ratio < output_aspect_ratio) {
        // Compute left and right padding.
        letterbox_pads[0] = (1.f - input_aspect_ratio / output_aspect_ratio) / 2.f;
        letterbox_pads[2] = letterbox_pads[0];
    } else if (output_aspect_ratio < input_aspect_ratio) {
        // Compute top and bottom padding.
        letterbox_pads[1] = (1.f - output_aspect_ratio / input_aspect_ratio) / 2.f;
        letterbox_pads[3] = letterbox_pads[1];
    }

    if (input_height != target_height || input_width !=target_width) {
        const float scale = std::min(static_cast<float>(target_width) / input_width,
                                     static_cast<float>(target_height) / input_height);
        const int resized_width  = std::round(input_width * scale);
        const int resized_height = std::round(input_height * scale);

        // TODO: we should use INTER_AREA when scale<1.0
        auto interp_mode = scale < 1.0f ? TNNInterpLinear : TNNInterpLinear;
        DimsVector intermediate_shape = {1, 3, resized_height, resized_width};
        auto intermediate_mat = std::make_shared<Mat>(cropped_mat->GetDeviceType(), mat->GetMatType(), intermediate_shape);
        auto status = Resize(cropped_mat, intermediate_mat, interp_mode);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);

        const int top    = (target_height - resized_height) / 2;
        const int bottom = (target_height - resized_height) - top;
        const int left   = (target_width  - resized_width) / 2;
        const int right  = (target_width  - resized_width) - left;

        auto input_mat = std::make_shared<Mat>(intermediate_mat->GetDeviceType(), mat->GetMatType(), target_dims);
        status = CopyMakeBorder(intermediate_mat, input_mat, top, bottom, left, right, TNNBorderConstant);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);

        return input_mat;
    }
    return cropped_mat;
}

MatConvertParam BlazePoseLandmark::GetConvertParamForInput(std::string tag) {
    MatConvertParam param;
    param.scale = {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0};
    param.bias  = {0.0,         0.0,         0.0,         0.0};
    return param;
}

std::shared_ptr<TNNSDKOutput> BlazePoseLandmark::CreateSDKOutput() {
    return std::make_shared<BlazePoseLandmarkOutput>();
}

Status BlazePoseLandmark::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto option = dynamic_cast<BlazePoseLandmarkOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                           Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    auto output = dynamic_cast<BlazePoseLandmarkOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
    Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));

    auto landmarks = output->GetMat("ld_3d");
    auto pose_presence_mat  = output->GetMat("output_poseflag");
    RETURN_VALUE_ON_NEQ(!landmarks, false,
                           Status(TNNERR_PARAM_ERR, "landmarks mat is nil"));
    RETURN_VALUE_ON_NEQ(!pose_presence_mat, false,
                           Status(TNNERR_PARAM_ERR, "pose_presence mat is nil"));
    float pose_presence_score = static_cast<float *>(pose_presence_mat->GetData())[0];
    if (pose_presence_score > option->pose_presence_threshold) {
        // generate roi according to landmarks
        roi_from_prev_frame = true;
    } else {
        // use pose_detection for the next frame
        roi_from_prev_frame = false;
    }
    std::vector<BlazePoseInfo> detects;
    ProcessLandmarks(*(landmarks.get()), detects);
    RemoveLetterBoxAndProjection(detects);
    //SmoothingLandmarks(detects);
    if (roi_from_prev_frame) {
        // generate roi for the next frame
        KeyPoints2RoI(detects[0].key_points_3d, this->roi_option);
    }
    SmoothingLandmarks(detects);
    DeNormalize(detects);
    // upper body landmark model only have 25 points for show
    detects[0].key_points_3d.resize(num_visible_landmarks);
    output->body_list.push_back(std::move(detects[0]));

    return status;
}

void BlazePoseLandmark::ProcessLandmarks(Mat& landmark_mat, std::vector<BlazePoseInfo>& detects) {
    detects.clear();

    const auto& input_dims = GetInputShape();
    const int input_height = input_dims[2];
    const int input_width  = input_dims[3];

    const float* landmark_data = static_cast<float *>(landmark_mat.GetData());
    const int num_dimensions = DimsVectorUtils::Count(landmark_mat.GetDims()) / num_landmarks;

    BlazePoseInfo info;
    info.key_points_3d.resize(num_landmarks);
    for(int i=0; i<num_landmarks; ++i) {
        int offset = i * num_dimensions;
        float x = landmark_data[offset + 0] / input_width;
        float y = landmark_data[offset + 1] / input_height;
        float z = landmark_data[offset + 2] / input_width;
        //float visibility = landmark_data[offset + 3];
        info.key_points_3d[i] = std::make_tuple(x, y, z);
    }
    info.lines = this->lines;
    detects.push_back(std::move(info));
}

void BlazePoseLandmark::RemoveLetterBoxAndProjection(std::vector<BlazePoseInfo>& detects) {
    const float left = letterbox_pads[0];
    const float top  = letterbox_pads[1];
    const float left_and_right = letterbox_pads[0] + letterbox_pads[2];
    const float top_and_bottom = letterbox_pads[1] + letterbox_pads[3];

    for(auto& lm3d: detects[0].key_points_3d) {
        // remove letterbox
        float x = (std::get<0>(lm3d) - left) / (1.0f - left_and_right);
        float y = (std::get<1>(lm3d) - top) / (1.0f - top_and_bottom);
        float z = std::get<2>(lm3d) / (1.0f - left_and_right);  // scale z coordinate as X.
        // projection
        x = x - 0.5f;
        y = y - 0.5f;
        float angle = roi.rotation;
        x = std::cos(angle) * x - std::sin(angle) * y;
        y = std::sin(angle) * x + std::cos(angle) * y;

        x = x * roi.width  + roi.x_center;
        y = y * roi.height + roi.y_center;
        z = z * roi.width;  // scale z coordinate as X.

        lm3d = std::make_tuple(x, y, z);
    }
}

void BlazePoseLandmark::DeNormalize(std::vector<BlazePoseInfo>& detects) {
    const int src_height = origin_input_shape[2];
    const int src_width  = origin_input_shape[3];

    for(auto& lm3d: detects[0].key_points_3d) {
        float x = std::get<0>(lm3d) * src_width;
        float y = std::get<1>(lm3d) * src_height;
        float z = std::get<2>(lm3d) * src_width;  // scale z coordinate as X.
        lm3d = std::make_tuple(x, y, z);
    }
    detects[0].image_height = src_height;
    detects[0].image_width  = src_width;
}

void BlazePoseLandmark::SmoothingLandmarks(std::vector<BlazePoseInfo> &detects) {
    auto image_size = std::make_pair(this->origin_input_shape[2], this->origin_input_shape[3]);
    std::vector<KeyPoint3D> out_landmarks;
    landmark_filter->Apply(detects[0].key_points_3d, image_size, Now(), &out_landmarks);
    if (out_landmarks.size() > 0) {
        detects[0].key_points_3d = out_landmarks;
    }
}

/*
 Generate a RoI for the blazepose landmark model
 This method corredponds to AlignmentPointsRectsCalculator+RectTransformationCalculator in mediapipe.
 By setting keypoints_idx, we could merge SplitNormalizedLandmarkListCalculator
 and LandmarksToDetectionCalculator together.
*/
// generate roi from the blazepose detection model
void BlazePoseLandmark::Detection2RoI(BlazePoseInfo& detect, const RoIGenOptions& option) {
    roi_from_prev_frame = false;
    auto& keypoints_2d = detect.key_points;
    return KeyPoints2RoI(keypoints_2d, option);
}

}
