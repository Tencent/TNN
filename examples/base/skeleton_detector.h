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

#ifndef TNN_EXAMPLES_BASE_SKELETON_DETECTOR_H_
#define TNN_EXAMPLES_BASE_SKELETON_DETECTOR_H_

#include "tnn_sdk_sample.h"
#include "landmark_smoothing_filter.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

namespace TNN_NS {

typedef ObjectInfo SkeletonInfo;

class SkeletonDetectorInput : public TNNSDKInput {
public:
    SkeletonDetectorInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~SkeletonDetectorInput(){}
};

/*
 the output of the skeleton model is a list of 2d points:
 point0: nose
 point1: left  eye
 point2: right eye
 point3: left  ear
 point4: right ear
 point5: left  shoulder
 point6: right shoulder
 point7: left  elbow
 point8: right elbow
 point9: left  wrist
 point10:right wrist
 point11:left  hip
 point12:right hip
 point13:left  knee
 point14:right knee
 point15:left  ankle
 point16:right ankle
 */

class SkeletonDetectorOutput : public TNNSDKOutput {
public:
    SkeletonDetectorOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~SkeletonDetectorOutput() {};
    SkeletonInfo keypoints;
    std::vector<float> confidence_list;
    std::vector<bool> detected;
};

class SkeletonDetectorOption : public TNNSDKOption {
public:
    SkeletonDetectorOption() {}
    virtual ~SkeletonDetectorOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    float min_threshold = 0.15;
    int fps = 20;
};

class SkeletonDetector : public TNNSDKSample {
public:
    virtual ~SkeletonDetector() {};
    
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);
    
private:
    void GenerateSkeleton(SkeletonDetectorOutput* output, std::shared_ptr<TNN_NS::Mat> heatmap,
                          float threshold);
    void SmoothingLandmarks(SkeletonDetectorOutput* output);
    void DeNormalize(SkeletonDetectorOutput* output);
    // the input mat size
    int orig_input_width;
    int orig_input_height;
    std::vector<SkeletonInfo> history;
    // lines for skeleton model:
    std::vector<std::pair<int, int>> lines = {
        {0, 1},
        {0, 2},
        {1, 3},
        {2, 4},
        {5, 6},
        {5, 7},
        {5, 11},
        {6, 8},
        {6, 12},
        {7, 9},
        {8, 10},
        {11,12},
        {11,13},
        {12,14},
        {13,15},
        {14,16}
    };
    // landmark filtering options
    const int window_size = 5;
    const float velocity_scale = 10.0;
    const float min_allowed_object_scale = 1e-6;
    std::shared_ptr<VelocityFilter> landmark_filter;
};

}

#endif // TNN_EXAMPLES_BASE_SKELETON_DETECTOR_H_

