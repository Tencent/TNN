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

#ifndef TNN_EXAMPLES_BASE_OCR_ANGLE_PREDICTOR_H_
#define TNN_EXAMPLES_BASE_OCR_ANGLE_PREDICTOR_H_

#include "tnn_sdk_sample.h"

#include <memory>
#include <string>
#include <vector>
#include <array>

namespace TNN_NS {

class OCRAnglePredictorOutput : public TNNSDKOutput {
public:
    OCRAnglePredictorOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~OCRAnglePredictorOutput();
    int index;
    float score;
};

class OCRAnglePredictor : public TNN_NS::TNNSDKSample {
public:
    ~OCRAnglePredictor();
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);
    
    // Process angles from same image
    void ProcessAngles(std::vector<std::shared_ptr<TNNSDKOutput>>& angles);
    bool DoAngle() { return do_angle_; }
    
private:
    bool do_angle_   = true;
    bool most_angle_ = true;
    int dst_width_   = 192;
    int dst_height_  = 32;
};

}
#endif // TNN_EXAMPLES_BASE_OCR_ANGLE_PREDICTOR_H_

