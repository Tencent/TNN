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

#ifndef TNN_EXAMPLES_BASE_BROADCAST_FEATURE_H_
#define TNN_EXAMPLES_BASE_BROADCAST_FEATURE_H_

#include "tnn_sdk_sample.h"
#include <memory>
#include <string>
#include <vector>

#if HAS_OPENCV

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp> // feature extraction
#include <opencv2/core/persistence.hpp> // FileStorage

namespace TNN_NS {


class BroadcastTextMatcherOutput : public TNNSDKOutput {
public:
    BroadcastTextMatcherOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~BroadcastTextMatcherOutput();
    int text_id;
    std::string text;
};

class BroadcastTextMatcherOption : public TNNSDKOption {
public:
    BroadcastTextMatcherOption() {}
    virtual ~BroadcastTextMatcherOption() {}
    // int num_threads = 1;
    std::string feature_path;
};

class BroadcastTextMatcher : public TNN_NS::TNNSDKSample {
public:
    ~BroadcastTextMatcher();
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                    std::string name = kTNNSDKDefaultName);
    // virtual cv::Mat ProcessSDKInputMat(std::shared_ptr<Mat> input_mat);
    
private:
    int TextMatch(cv::Mat image, float threshold, float dist_ratio);
    int Match(std::vector<cv::DMatch>& matches, cv::Mat des1, cv::Mat des2, float dist_ratio);


    // std::vector<std::vector<cv::KeyPoint>> keypoints_;
    std::vector<cv::Mat> descriptors_;
    std::vector<int> ids_;
    std::vector<std::string> texts_;
    std::vector<int> words_;
    cv::Ptr<cv::SIFT> detector_;
    cv::Ptr<cv::BFMatcher> matcher_;
    
    int word_norm_ = 14;
    float text_thresh_ = 6.0f;
    float text_dist_ratio_ = 0.7f;
};

}

#endif // HAS_OPENCV

#endif // TNN_EXAMPLES_BASE_BROADCAST_FEATURE_H_
