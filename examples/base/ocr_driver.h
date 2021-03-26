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

#ifndef TNN_EXAMPLES_BASE_OCR_DRIVER_H_
#define TNN_EXAMPLES_BASE_OCR_DRIVER_H_

#include <algorithm>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <array>

#include "tnn_sdk_sample.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

#if HAS_OPENCV
#include "opencv2/core/core.hpp"
#endif

namespace TNN_NS{

class OCROutput : public TNNSDKOutput {
public:
    OCROutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~OCROutput() {};
    //std::vector<float> scores;
    int image_height;
    int image_width;
    std::vector<std::string> texts;
    std::vector<std::pair<float, float>> box;
    float angle;
};

class OCRDriver : public TNN_NS::TNNSDKComposeSample {
public:
    virtual ~OCRDriver() {}
    
    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);
    
    virtual Status Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks);
    
    void SetBlobDumpDir(const char *dump_dir);

protected:
#if HAS_OPENCV
    Status MatToTNNMat(const cv::Mat& mat, std::shared_ptr<Mat>& tnn_mat, bool try_share_data);
#endif
    std::shared_ptr<TNNSDKSample> textbox_detector_;
    std::shared_ptr<TNNSDKSample> angle_predictor_;
    std::shared_ptr<TNNSDKSample> text_recognizer_;
};

}

#endif // TNN_EXAMPLES_BASE_OCR_DRIVER_H_

