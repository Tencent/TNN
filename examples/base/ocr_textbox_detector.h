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

#ifndef TNN_EXAMPLES_BASE_OCR_TEXTBOX_DETECTOR_H_
#define TNN_EXAMPLES_BASE_OCR_TEXTBOX_DETECTOR_H_

#include "tnn_sdk_sample.h"

#if HAS_OPENCV

#include "opencv2/core/core.hpp"

#include <memory>
#include <string>
#include <vector>
#include <array>

namespace TNN_NS {

struct TextBox {
    std::vector<cv::Point> box_points;
    // box points coresponding to sdk input
    std::vector<cv::Point> box_points_input;
    float score;
    int image_width;
    int image_height;
};

struct ScaleParam {
    int srcWidth;
    int srcHeight;
    int dstWidth;
    int dstHeight;
    float ratioWidth;
    float ratioHeight;
};

class OCRTextboxDetectorOption : public TNNSDKOption {
public:
    OCRTextboxDetectorOption() {}
    virtual ~OCRTextboxDetectorOption() {}
    int padding = 50;
    float box_score_threshold = 0.6f;
    float scale_down_ratio    = 1.00f;
};

class OCRTextboxDetectorOutput : public TNNSDKOutput {
public:
    OCRTextboxDetectorOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~OCRTextboxDetectorOutput();
    std::vector<TextBox> text_boxes;
};

class OCRTextboxDetector : public TNN_NS::TNNSDKSample {
public:
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    ~OCRTextboxDetector();
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);
    
    cv::Mat& GetPaddedInput() { return padded_input_; }
    
private:
    int padding_  = 10;
    float box_score_thresh_ = 0.6f;
    float scale_down_ratio_ = 0.75f;

    float box_thresh_ = 0.3f;
    int max_size_ = 1024;
    float un_clip_ratio_ = 2.0f;
    int input_height_;
    int input_width_;
    cv::Mat padded_input_;
    ScaleParam scale_;
};

}
#endif  // HAS_OPENCV

#endif // TNN_EXAMPLES_BASE_OCR_TEXTBOX_DETECTOR_H_
