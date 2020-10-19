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

#ifndef TNN_EXAMPLES_BASE_BLAZEPOSE_DETECTOR_H_
#define TNN_EXAMPLES_BASE_BLAZEPOSE_DETECTOR_H_

#include "tnn_sdk_sample.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <array>

namespace TNN_NS {

typedef ObjectInfo BlazePoseInfo;

class BlazePoseDetectorInput : public TNNSDKInput {
public:
    BlazePoseDetectorInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~BlazePoseDetectorInput(){}
};

class BlazePoseDetectorOutput : public TNNSDKOutput {
public:
    BlazePoseDetectorOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~BlazePoseDetectorOutput() {};
    std::vector<BlazePoseInfo> body_list;
};

class BlazePoseDetectorOption : public TNNSDKOption {
public:
    BlazePoseDetectorOption() {}
    virtual ~BlazePoseDetectorOption() {}
    int input_width;
    int input_height;

    int num_thread = 1;
    float min_score_threshold = 0.5;
    float min_suppression_threshold = 0.3;
};

class BlazePoseDetector : public TNNSDKSample {
public:
    virtual ~BlazePoseDetector() {};

    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);
    std::shared_ptr<Mat> input_;

private:
    // struct for a detection anchor
    struct Anchor {
        // Encoded anchor box center.
        float x_center;
        float y_center;
        // Encoded anchor box height.
        float h;
        // Encoded anchor box width.
        float w;
    };
    // configs for generating anchors
    struct AnchorOptions {
        int num_layers = 4;
        float min_scale = 0.1484375;
        float max_scale = 0.75;
        float anchor_offset_x = 0.5;
        float anchor_offset_y = 0.5;
        std::vector<int> strides = {8, 16, 16, 16};
        std::vector<float> aspect_ratios = {1.0};
        float interpolated_scale_aspect_ratio = 1.0;
    };
    // configs for decoding score and boxes
    struct DeocdingOptions {
        int num_boxes = 896;
        int num_coords = 12;
        int num_classes = 1;
        int num_keypoints = 4;
        int keypoint_coord_offset = 4;
        int num_values_per_keypoint = 2;
        bool sigmoid_score = true;
        float score_clipping_thresh = 100;
        bool reverse_output_order = true;
    };

private:
    void GenerateAnchor(std::vector<Anchor>* anchors);
    void GenerateBBox(std::vector<BlazePoseInfo> &detects, Mat &scores, Mat &boxes,
                      float min_score_threshold);
    void DecodeBoxes(std::vector<float>& boxes, const float* raw_boxes);
    void DecodeScore(std::vector<float>& scores, std::vector<int>& classes, const float* raw_scores);
    void RemoveLetterBox(std::vector<BlazePoseInfo>& detects);
    // pads for remove latterbox from detection
    std::array<float, 4> letterbox_pads;

    AnchorOptions anchor_options;
    std::vector<Anchor> anchors;
    // box decoding config
    DeocdingOptions decode_options;
};

}

#endif // TNN_EXAMPLES_BASE_BLAZEPOSE_DETECTOR_H_

