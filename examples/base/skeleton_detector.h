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
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

namespace TNN_NS {

typedef ObjectInfo SkeletonInfo;

class SkeletonDetectorInput : public TNNSDKInput {
public:
    SkeletonDetectorInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~SkeletonDetectorInput(){}
};

class SkeletonDetectorOutput : public TNNSDKOutput {
public:
    SkeletonDetectorOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~SkeletonDetectorOutput() {};
    std::vector<SkeletonInfo> keypoint_list;
};

class SkeletonDetectorOption : public TNNSDKOption {
public:
    SkeletonDetectorOption() {}
    virtual ~SkeletonDetectorOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    float min_threshold = 0.15;
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
    void GenerateSkeleton(std::vector<SkeletonInfo> &skeleton, std::shared_ptr<TNN_NS::Mat> heatmap, float threshold);
    TNN_NS::Status GaussianBlur(std::shared_ptr<TNN_NS::Mat>src, std::shared_ptr<TNN_NS::Mat>,
                      int kernel_h, int kernel_w,
                      float sigma_x, float sigma_y);
    // the input mat size
    int orig_input_width;
    int orig_input_height;
};

}

#endif // TNN_EXAMPLES_BASE_SKELETON_DETECTOR_H_

