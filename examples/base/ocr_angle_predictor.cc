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

#include "ocr_angle_predictor.h"

#if HAS_OPENCV

#include "opencv2/core/mat.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <cmath>

namespace TNN_NS {

OCRAnglePredictorOutput::~OCRAnglePredictorOutput() {}

MatConvertParam OCRAnglePredictor::GetConvertParamForInput(std::string name) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5, 0.0};
    input_convert_param.bias  = {-1.0,        -1.0,        -1.0,       0.0};
    // model requires RGB input
    input_convert_param.reverse_channel = false;
    
    return input_convert_param;
}

std::shared_ptr<Mat> OCRAnglePredictor::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
                                                                   std::string name) {
    Status status = TNN_OK;

    // 0) copy if necessary
    bool need_copy = false;
    DeviceType origin_dev = input_mat->GetDeviceType();
    if (input_mat->GetDeviceType() != DEVICE_ARM && device_type_ == DEVICE_ARM) {
        need_copy = true;
        auto input_arm_mat = std::make_shared<Mat>(DEVICE_ARM, input_mat->GetMatType(),
                                                   input_mat->GetDims());
        status = Copy(input_mat, input_arm_mat);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);
        input_mat = input_arm_mat;
    }

    // 1) TNN::Mat to opencv Mat
    int img_height = input_mat->GetHeight();
    int img_width  = input_mat->GetWidth();
    void *pixel = input_mat->GetData();
    cv::Mat cv_src(img_height, img_width, CV_8UC4, pixel);

    // 2) resize
    float scale = static_cast<float>(dst_height_) / img_height;
    int angleWidth = static_cast<int>(img_width * scale);
    if (scale != 1) {
        cv::Mat resized_src;
        cv::resize(cv_src, resized_src, cv::Size(angleWidth, dst_height_));
        cv_src = resized_src;
    }
    cv::Mat srcFit = cv::Mat(dst_height_, dst_width_, CV_8UC4, cv::Scalar(255, 255, 255));
    if (angleWidth < dst_width_) {
        cv::Rect rect(0, 0, cv_src.cols, cv_src.rows);
        cv_src.copyTo(srcFit(rect));
    } else {
        cv::Rect rect(0, 0, dst_width_, dst_height_);
        cv_src(rect).copyTo(srcFit);
    }

    // 3) cv::Mat to TNN::Mat
    int input_height = srcFit.rows;
    int input_width  = srcFit.cols;
    auto input_shape = input_mat->GetDims();
    input_shape[2] = input_height;
    input_shape[3] = input_width;
    
    std::shared_ptr<Mat> result_mat = nullptr;
    if (need_copy) {
        auto input_arm_mat = std::make_shared<Mat>(DEVICE_ARM, input_mat->GetMatType(),
                                                   input_shape, srcFit.data);
        result_mat = std::make_shared<Mat>(origin_dev, input_mat->GetMatType(), input_shape);
        status = Copy(input_arm_mat, result_mat);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);
    } else {
        result_mat = std::make_shared<Mat>(input_mat->GetDeviceType(), N8UC4, input_shape);
        memcpy(result_mat->GetData(), srcFit.data, sizeof(uint8_t)*DimsVectorUtils::Count(input_shape));
    }

    return result_mat;
}

std::shared_ptr<TNNSDKOutput> OCRAnglePredictor::CreateSDKOutput() {
    return std::make_shared<OCRAnglePredictorOutput>();
}

Status OCRAnglePredictor::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto output = dynamic_cast<OCRAnglePredictorOutput *>(output_.get());
    auto output_mat = output->GetMat("out");
    const auto output_count = DimsVectorUtils::Count(output_mat->GetDims());
    float *output_data = static_cast<float *>(output_mat->GetData());

    float max_score = output_data[0];
    int max_idx = 0;
    for(int i=1; i<output_count; ++i) {
        if (output_data[i] > max_score) {
            max_score = output_data[i];
            max_idx   = i;
        }
    }
    output->index = max_idx;
    output->score = max_score;

    return status;
}

void OCRAnglePredictor::ProcessAngles(std::vector<std::shared_ptr<TNNSDKOutput>>& angles) {
    if (!do_angle_ || !most_angle_)
        return ;

    float index_sum = 0;
    float half_percent = angles.size() / 2.0f;
    for(const auto& output : angles) {
        auto angle = dynamic_cast<OCRAnglePredictorOutput *>(output.get());
        index_sum += angle->index;
    }
    int voted_angle = index_sum >= half_percent;
    for(auto& output : angles) {
        auto angle = dynamic_cast<OCRAnglePredictorOutput *>(output.get());
        angle->index = voted_angle;
    }
}

OCRAnglePredictor::~OCRAnglePredictor() {}

}

#endif// HAS_OPENCV
