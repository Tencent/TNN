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

#ifndef TNN_EXAMPLES_BASE_HAND_TRACKING_H_
#define TNN_EXAMPLES_BASE_HAND_TRACKING_H_

#include <algorithm>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <array>

#include "tnn_sdk_sample.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS{

class HandTrackingInput : public TNNSDKInput {
public:
    HandTrackingInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~HandTrackingInput(){}
};

class HandTrackingOutput : public TNNSDKOutput {
public:
    HandTrackingOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~HandTrackingOutput() {};
    std::vector<ObjectInfo> hand_list;
};

class HandTrackingOption : public TNNSDKOption {
public:
    HandTrackingOption() {}
    virtual ~HandTrackingOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;

    float hand_presence_threshold = 0.5;
};

class HandTracking : public TNN_NS::TNNSDKSample {
public:
    virtual ~HandTracking() {}
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat, std::string name = kTNNSDKDefaultName);
   // virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);
private:
    void SetHandRegion(int x1, int y1, int x2, int y2);
    // hand region set by the detection model or in previous frame
    float x1_;
    float y1_;
    float x2_;
    float y2_;
    // whether valid hand in this frame
    bool valid_hand_in_prev_frame_ = false;
};

}

#endif // TNN_EXAMPLES_BASE_HAND_TRACKING_H_
