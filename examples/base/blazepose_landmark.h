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

#ifndef TNN_EXAMPLES_BASE_BLAZEPOSE_LANDMARK_H_
#define TNN_EXAMPLES_BASE_BLAZEPOSE_LANDMARK_H_

#include "tnn_sdk_sample.h"
#include "blazepose_detector.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <array>

namespace TNN_NS {

struct ROIRect {
    // rotate angle in radian
    float rotation;
    float width;
    float height;
    float x_center;
    float y_center;
};

class BlazePoseLandmarkInput : public TNNSDKInput {
public:
    BlazePoseLandmarkInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~BlazePoseLandmarkInput(){}
};

class BlazePoseLandmarkOutput : public TNNSDKOutput {
public:
    BlazePoseLandmarkOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~BlazePoseLandmarkOutput() {};
    std::vector<BlazePoseInfo> body_list;
};

class BlazePoseLandmarkOption : public TNNSDKOption {
public:
    BlazePoseLandmarkOption() {}
    virtual ~BlazePoseLandmarkOption() {}
    int input_width;
    int input_height;

    int num_thread = 1;
    // threshold for the pose presenting in landmark model
    float pose_presence_threshold = 0.5;
};

class BlazePoseLandmark : public TNNSDKSample {
public:
    virtual ~BlazePoseLandmark() {};

    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                    std::string name = kTNNSDKDefaultName);
    struct RoIGenOptions {
        // which keypoints will be used to generate roi
        int keypoints_start_idx;
        int keypoints_end_idx;
        // the target angle, in degrees
        float rotation_target_angle;
        float scale_x;
        float scale_y;
    };
    void Detection2RoI(BlazePoseInfo& detect, const RoIGenOptions& option);
    bool NeedPoseDetection() {
        return !roi_from_prev_frame;
    }
    void SetOrigianlInputShape(int height, int width) {
        if (this->origin_input_shape.size() <= 0)
            this->origin_input_shape.resize(4, 0);
        this->origin_input_shape[2] = height;
        this->origin_input_shape[3] = width;
    }
private:
    void GetCropMatrix(const ROIRect& roi, float trans_mat[2][3], std::vector<float>& target_size);
    void ProcessLandmarks(Mat& landmark_mat, std::vector<BlazePoseInfo>& detects);
    void RemoveLetterBoxAndProjection(std::vector<BlazePoseInfo>& detects);
    void DeNormalize(std::vector<BlazePoseInfo>& detects);
    // alias for 2d keypoints
    using Keypoints2D = std::pair<float, float>;
    using Keypoints3D = triple<float,float,float>;
    template <typename KeyPointType>
    void KeyPoints2RoI(std::vector<KeyPointType>& key_points, const RoIGenOptions& option);
    // the orignal input shape
    DimsVector origin_input_shape;
    // the roi
    ROIRect roi;
    // if the roi comes from previous frame
    bool roi_from_prev_frame = false;
    // option used to generate roi for the next frame
    RoIGenOptions roi_option = {
        25,
        26,
        90.0f,
        1.5f,
        1.5f
    };
    // pads for remove latterbox from detection
    std::array<float, 4> letterbox_pads;
    int num_landmarks = 31;
};

}

#endif // TNN_EXAMPLES_BASE_BLAZEPOSE_LANDMARK_H_


