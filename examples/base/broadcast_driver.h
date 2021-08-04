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

#ifndef TNN_EXAMPLES_BASE_BROADCAST_DRIVER_H_
#define TNN_EXAMPLES_BASE_BROADCAST_DRIVER_H_

#include "tnn_sdk_sample.h"

#if HAS_OPENCV

#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

#include "opencv2/core/mat.hpp"

#include <algorithm>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <array>

namespace TNN_NS{

struct HeroSide {
    int x1;
    int y1;
    int x2;
    int y2;
    cv::Mat image;
};

class BroadcastOutput : public TNNSDKOutput {
public:
    BroadcastOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~BroadcastOutput() {};
    //std::vector<float> scores;
    int image_height;
    int image_width;
    std::vector<std::string> texts;
    std::vector<std::pair<float, float>> box;
    std::vector<ObjectInfo> heros;
    float angle;
};

class BroadcastDriver : public TNN_NS::TNNSDKComposeSample {
public:
    virtual ~BroadcastDriver() {}
    
    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);
    
    virtual Status Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks);

    virtual bool hideTextBox();

protected:
    Status MatToTNNMat(const cv::Mat& mat, std::shared_ptr<Mat>& tnn_mat, bool try_share_data);

    std::shared_ptr<TNNSDKSample> textbox_detector_;
    std::shared_ptr<TNNSDKSample> text_matcher_;
    std::shared_ptr<TNNSDKSample> hero_detector_;
};

}

#endif // HAS_OPENCV

#endif // TNN_EXAMPLES_BASE_BROADCAST_DRIVER_H_

