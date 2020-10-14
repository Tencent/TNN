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

namespace TNN_NS {

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

    return status;
}

std::shared_ptr<Mat> BlazePoseLandmark::ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                                   std::string name) {
    auto target_dims   = GetInputShape(name);
    auto target_height = target_dims[2];
    auto target_width  = target_dims[3];
    // TODO: step1: cropping roi from input image
    auto cropped_mat = mat;

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
        DimsVector intermediate_shape = {1, 4, resized_height, resized_width};
        auto intermediate_mat = std::make_shared<Mat>(DEVICE_ARM, N8UC4, intermediate_shape);
        auto status = Resize(cropped_mat, intermediate_mat, interp_mode);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);

        const int top    = (target_height - resized_height) / 2;
        const int bottom = (target_height - resized_height) - top;
        const int left   = (target_width  - resized_width) / 2;
        const int right  = (target_width  - resized_width) - left;

        auto input_mat = std::make_shared<Mat>(DEVICE_ARM, N8UC4, target_dims);
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
    auto pose_presence_mat  = output->GetMat("regressors");
    RETURN_VALUE_ON_NEQ(!landmarks, false,
                           Status(TNNERR_PARAM_ERR, "landmarks mat is nil"));
    RETURN_VALUE_ON_NEQ(!pose_presence_mat, false,
                           Status(TNNERR_PARAM_ERR, "pose_presence mat is nil"));
    float pose_presence_score = static_cast<float *>(pose_presence_mat->GetData())[0];
    if (pose_presence_score > option->pose_presence_threshold) {
        // TODO: generate roi according to landmarks
    }

    std::vector<BlazePoseInfo> detects;
    ProcessLandmarks(*(landmarks.get()), detects);
    RemoveLetterBoxAndProjection(detects);
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

}


