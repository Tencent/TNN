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
#include <sys/time.h>
#include <cmath>
#include <fstream>
#include <cstring>
#include <algorithm>

namespace TNN_NS {

template <>
void BlazePoseLandmark::KeyPoints2RoI(std::vector<Keypoints2D>& key_points, const RoIGenOptions& option) {
    constexpr double PI = 3.14159265358979323846264338327950288;
    const int start_kp_idx = option.keypoints_start_idx;
    const int end_kp_idx = option.keypoints_end_idx;
    const float target_angle = option.rotation_target_angle / 180.0 * PI;

    const int input_height = origin_input_shape[2];
    const int input_width  = origin_input_shape[3];

    float x_center = 0;
    float y_center = 0;
    float x_scale  = 0;
    float y_scale  = 0;

    x_center = key_points[start_kp_idx].first * input_width;
    y_center = key_points[start_kp_idx].second * input_height;
    x_scale  = key_points[end_kp_idx].first * input_width;
    y_scale  = key_points[end_kp_idx].second * input_height;

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

template <>
void BlazePoseLandmark::KeyPoints2RoI(std::vector<Keypoints3D>& key_points, const RoIGenOptions& option) {
    constexpr double PI = 3.14159265358979323846264338327950288;
    const int start_kp_idx = option.keypoints_start_idx;
    const int end_kp_idx = option.keypoints_end_idx;
    const float target_angle = option.rotation_target_angle / 180.0 * PI;

    const int input_height = origin_input_shape[2];
    const int input_width  = origin_input_shape[3];

    float x_center = 0;
    float y_center = 0;
    float x_scale  = 0;
    float y_scale  = 0;

    x_center = std::get<0>(key_points[start_kp_idx]) * input_width;
    y_center = std::get<1>(key_points[start_kp_idx]) * input_height;
    x_scale  = std::get<0>(key_points[end_kp_idx]) * input_width;
    y_scale  = std::get<1>(key_points[end_kp_idx]) * input_height;

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

    /*
     roi from detects:
     center:(x, y)0.613132,0.243556
     (height, width))0.66414,0.88206
     rotation:-0.0125525

     roi from landmarks:
     center:(x, y)0.608104,0.257316
     (height, width))0.685761,0.910776
     rotation:0.0741528
     */
    roi.x_center = x_center;
    roi.y_center = y_center;
    roi.width  = width  * option.scale_x;
    roi.height = height * option.scale_y;
    roi.rotation = rotation;
}

void BlazePoseLandmark::GetCropMatrix(const ROIRect& roi, float trans_mat[2][3], std::vector<float>&target_size) {
    // (1) compute rotation;
    float rotate = roi.rotation;

    // (2) compute resize scale
    auto input_shape = GetInputShape();
    const int input_width  = input_shape[3];
    const int input_height = input_shape[2];
    float scale = std::min({1.0f, input_height/roi.height, input_width/roi.width});
    float target_width  = scale * roi.width;
    float target_height = scale * roi.height;
    trans_mat[0][0] = scale * std::cos(rotate);
    trans_mat[0][1] = scale * std::sin(rotate);
    trans_mat[1][0] = -scale* std::sin(rotate);
    trans_mat[1][1] = scale * std::cos(rotate);

    // (3) compute offset
    float trans_center_x = roi.x_center * trans_mat[0][0] + roi.y_center * trans_mat[0][1];
    float trans_center_y = roi.x_center * trans_mat[1][0] + roi.y_center * trans_mat[1][1];
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

    auto input_dims = GetInputShape();
    option->input_height = input_dims[2];
    option->input_width  = input_dims[3];
    roi_from_prev_frame = false;

    return status;
}

std::shared_ptr<Mat> BlazePoseLandmark::ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                                   std::string name) {
    // save the origianl input shape
    origin_input_shape = mat->GetDims();
    const int src_height = origin_input_shape[2];
    const int src_width  = origin_input_shape[3];

    auto target_dims   = GetInputShape(name);
    auto target_height = target_dims[2];
    auto target_width  = target_dims[3];

    // step1: cropping roi from input image
    float normalized_width  = this->roi.width;
    float normalized_height = this->roi.height;
    float normalized_x_center = this->roi.x_center;
    float normalized_y_center = this->roi.y_center;
    float rotation = this->roi.rotation;
    float crop_width  = std::round(normalized_width  * src_width);
    float crop_height = std::round(normalized_height * src_height);
    int x_center = std::round(normalized_x_center * src_width);
    int y_center = std::round(normalized_y_center * src_height);

    float transMat[2][3];
    std::vector<float> cropSize;
    ROIRect rect;
    rect.rotation = rotation;
    rect.height = crop_height;
    rect.width = crop_width;
    rect.x_center = x_center;
    rect.y_center = y_center;
    GetCropMatrix(rect, transMat, cropSize);
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

        // TODO: we should use INTER_AREA when scale<1.0, use INTER_LINEAR for now, as TNN does not support INTER_AREA
        auto interp_mode = scale < 1.0f ? TNNInterpLinear : TNNInterpLinear;
        DimsVector intermediate_shape = {1, 3, resized_height, resized_width};
        auto intermediate_mat = std::make_shared<Mat>(cropped_mat->GetDeviceType(), mat->GetMatType(), intermediate_shape);
        auto status = Resize(cropped_mat, intermediate_mat, interp_mode);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);

        const int top    = (target_height - resized_height) / 2;
        const int bottom = (target_height - resized_height) - top;
        const int left   = (target_width  - resized_width) / 2;
        const int right  = (target_width  - resized_width) - left;

        auto input_mat = std::make_shared<Mat>(cropped_mat->GetDeviceType(), mat->GetMatType(), target_dims);
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
    //TODO: ensure mediapipe requires RGB or BGR
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
    /*
     landmark[0]:(0.585276, 0.176491, 0.001474)
     landmark[1]:(0.581927, 0.160643, 0.001763)
     landmark[2]:(0.576502, 0.160856, -0.001101)
     landmark[3]:(0.571206, 0.161458, 0.002808)
     landmark[4]:(0.601131, 0.159893, -0.005054)
     landmark[5]:(0.608473, 0.159413, -0.002445)
     landmark[6]:(0.614851, 0.158913, 0.004186)
     landmark[7]:(0.572033, 0.171937, 0.000685)
     landmark[8]:(0.629830, 0.168300, 0.001106)
     landmark[9]:(0.581822, 0.192919, -0.001152)
     landmark[10]:(0.600319, 0.191274, 0.004665)
     landmark[11]:(0.519896, 0.258741, -0.032623)
     landmark[12]:(0.698915, 0.258369, 0.025607)
     landmark[13]:(0.501904, 0.368605, 0.121646)
     landmark[14]:(0.721328, 0.365689, 0.255914)
     landmark[15]:(0.528432, 0.401184, 0.590652)
     landmark[16]:(0.725253, 0.418987, 0.541670)
     landmark[17]:(0.531659, 0.411886, -0.003565)
     landmark[18]:(0.730380, 0.435008, -0.002912)
     landmark[19]:(0.535556, 0.401903, -0.004019)
     landmark[20]:(0.723725, 0.430686, 0.002975)
     landmark[21]:(0.539524, 0.402973, 0.002815)
     landmark[22]:(0.716350, 0.427519, -0.002361)
     landmark[23]:(0.531858, 0.408288, 0.226552)
     landmark[24]:(0.654916, 0.411333, 0.313479)
     landmark[25]:(0.608104, 0.257316, -0.000006)
     landmark[26]:(0.630596, 0.029357, 0.000003)
     landmark[27]:(0.538684, 0.405472, 0.000012)
     landmark[28]:(0.530957, 0.400556, 0.000004)
     landmark[29]:(0.713909, 0.430072, 0.000002)
     landmark[30]:(0.716196, 0.418809, -0.000010)
     */
    if (roi_from_prev_frame) {
        // generate roi for the next frame
        KeyPoints2RoI(detects[0].key_points_3d, this->roi_option);
    }
    SmoothingLandmarks(detects);
    DeNormalize(detects);
    
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
        // TODO: devan do we need visibility?
        //float visibility = landmark_data[offset + 3];
        info.key_points_3d[i] = std::make_tuple(x, y, z);
    }
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

        // TODO: devan do we need visibility?
        // Keep visibility as is.
        //new_landmark->set_visibility(landmark.visibility());
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
    constexpr float decay = 0.6f;
    
    using Point3D = std::tuple<float, float, float>;
    auto weighted_sum = [](Point3D& a, Point3D& b) {
        auto x = std::get<0>(a)*(1-decay) + std::get<0>(b)*decay;
        auto y = std::get<1>(a)*(1-decay) + std::get<1>(b)*decay;
        auto z = std::get<2>(a)*(1-decay) + std::get<2>(b)*decay;
        return std::make_tuple(x, y, z);
    };
    
    if (history.size() > 0) {
        // smoothing using history
        auto& cur_kp3d = detects[0].key_points_3d;
        auto& his_kp3d = history[0].key_points_3d;
        for(int i=0; i<cur_kp3d.size(); ++i) {
            cur_kp3d[i] = weighted_sum(cur_kp3d[i], his_kp3d[i]);
        }
        his_kp3d = cur_kp3d;
    } else {
        history.push_back(detects[0]);
    }
    return;
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
