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

#ifndef TNN_EXAMPLES_BASE_BLAZEFACE_DETECTOR_H_
#define TNN_EXAMPLES_BASE_BLAZEFACE_DETECTOR_H_

#include "tnn_sdk_sample.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

namespace TNN_NS {

typedef ObjectInfo BlazeFaceInfo;

class BlazeFaceDetectorInput : public TNNSDKInput {
public:
    BlazeFaceDetectorInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~BlazeFaceDetectorInput(){}
};

class BlazeFaceDetectorOutput : public TNNSDKOutput {
public:
    BlazeFaceDetectorOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~BlazeFaceDetectorOutput() {};
    std::vector<BlazeFaceInfo> face_list;
};

class BlazeFaceDetectorOption : public TNNSDKOption {
public:
    BlazeFaceDetectorOption() {}
    virtual ~BlazeFaceDetectorOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    float min_score_threshold = 0.75;
    float min_suppression_threshold = 0.3;
    std::string anchor_path;
};

class BlazeFaceDetector : public TNNSDKSample {
public:
    virtual ~BlazeFaceDetector() {};
    
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);
    
private:
    void GenerateBBox(std::vector<BlazeFaceInfo> &detects, Mat &scores, Mat &boxes,
                      int image_w, int image_h, float min_score_threshold);
    void BlendingNMS(std::vector<BlazeFaceInfo> &input, std::vector<BlazeFaceInfo> &output,
                     float min_suppression_threshold);
    
    std::vector<float> anchors;
    
    int num_anchors = 896;
    int detect_dims = 16;
    int num_keypoints = 6;
};

}

#endif // TNN_EXAMPLES_BASE_BLAZEFACE_DETECTOR_H_
