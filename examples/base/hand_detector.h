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

#ifndef TNN_EXAMPLES_BASE_HAND_DETECTOR_H_
#define TNN_EXAMPLES_BASE_HAND_DETECTOR_H_

#include "tnn_sdk_sample.h"
#include <memory>
#include <string>
#include <vector>

namespace TNN_NS {

class HandDetectorOutput : public TNNSDKOutput {
public:
    HandDetectorOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~HandDetectorOutput();
    std::vector<ObjectInfo> hands;
};

class HandDetectorOption : public TNNSDKOption {
public:
    HandDetectorOption() {}
    virtual ~HandDetectorOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;

    float conf_threshold = 0.3;
    float nms_threshold  = 0.01;
};


class HandDetector : public TNN_NS::TNNSDKSample {
public:
    ~HandDetector() {}

    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);
private:
    float GetGridX(int h, int w) {
        return w % this->grid_size;
    }
    float GetGridY(int h, int w) {
        return h % this->grid_size;
    }
    void GenerateBBox(Mat *x, Mat *y, Mat *h, Mat *w, Mat *conf, std::vector<ObjectInfo>& bboxes);
    void BlendingNMS(std::vector<ObjectInfo> &input, std::vector<ObjectInfo> &output);
    // detector configs
    float stride  = 16.0;
    int grid_size = 26.0;
    std::vector<float> anchor_w = {33.0/16.0, 62.0/16.0, 116.0/16.0};
    std::vector<float> anchor_h = {23.0/16.0, 45.0/16.0, 90.0/16.0};

    float conf_thresh;
    float nms_thresh;
};

}
#endif // TNN_EXAMPLES_BASE_HAND_DETECTOR_H_

