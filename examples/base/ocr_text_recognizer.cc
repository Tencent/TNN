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

#include "ocr_text_recognizer.h"

#if HAS_OPENCV

#include "opencv2/core/mat.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <cmath>


namespace TNN_NS {

OCRTextRecognizerOutput::~OCRTextRecognizerOutput() {}

Status OCRTextRecognizer::Init(std::shared_ptr<TNNSDKOption> option) {
    option->input_shapes.insert( {"input", DimsVector({1, 3, dst_height_, max_width_})} );
    // load vocabulary
    const auto& vocab_file_path = dynamic_cast<OCRTextRecognizerOption *>(option.get())->vocab_path;
    std::ifstream in(vocab_file_path.c_str());
    if (!in) {
        return Status(TNNERR_PARAM_ERR, "invalid vocabulary file path!");
    }
    std::string line;
    while(getline(in, line)) {
        vocabulary_.push_back(line);
    }
    if (vocabulary_.size() != vocab_size_) {
        return Status(TNNERR_PARAM_ERR, "invalid vocabulary file!");
    }
    return TNNSDKSample::Init(option);
}

MatConvertParam OCRTextRecognizer::GetConvertParamForInput(std::string name) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5, 0.0};
    input_convert_param.bias  = {-1.0,        -1.0,        -1.0,       0.0};
    // model requires RGB input
    input_convert_param.reverse_channel = false;
    
    return input_convert_param;
}

std::shared_ptr<Mat> OCRTextRecognizer::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
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
    if (scale != 1) {
        int dst_width = static_cast<int>(img_width * scale);
        cv::Mat resized_src;
        cv::resize(cv_src, resized_src, cv::Size(dst_width, dst_height_));
        cv_src = resized_src;
    }

    // 3) cv::Mat to TNN::Mat
    int input_height = cv_src.rows;
    int input_width  = cv_src.cols;
    auto input_shape = input_mat->GetDims();
    input_shape[2] = input_height;
    input_shape[3] = input_width;

    std::shared_ptr<Mat> result_mat = nullptr;
    if (need_copy) {
        auto input_arm_mat = std::make_shared<Mat>(DEVICE_ARM, input_mat->GetMatType(),
                                                   input_shape, cv_src.data);
        result_mat = std::make_shared<Mat>(origin_dev, input_mat->GetMatType(), input_shape);
        status = Copy(input_arm_mat, result_mat);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);
    } else {
        result_mat = std::make_shared<Mat>(input_mat->GetDeviceType(), N8UC4, input_shape);
        memcpy(result_mat->GetData(), cv_src.data, sizeof(uint8_t)*DimsVectorUtils::Count(input_shape));
    }

    // 4) reshape
    InputShapesMap input_shape_map;
    const auto input_name = GetInputNames()[0];
    input_shape[1] = 3;
    if (input_shape[3] > max_width_) {
        LOGE("invalid input: input width:%d is too large!\n", input_shape[3]);
        return nullptr;
    }
    input_shape_map.insert({input_name, input_shape});
    status = instance_->Reshape(input_shape_map);
    if (status != TNN_OK) {
        LOGE("instance Reshape failed in text recognizer\n");
        return nullptr;
    }

    return result_mat;
}

std::shared_ptr<TNNSDKOutput> OCRTextRecognizer::CreateSDKOutput() {
    return std::make_shared<OCRTextRecognizerOutput>();
}

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

Status OCRTextRecognizer::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto output = dynamic_cast<OCRTextRecognizerOutput *>(output_.get());
    auto output_mat = output->GetMat("out");
    const auto output_shape = output_mat->GetDims();
    float *output_data = static_cast<float *>(output_mat->GetData());
    
    const int seq_len = output_shape[0];
    const int vocab_len = output_shape[2];
    if (vocab_len != vocabulary_.size()) {
        return Status(TNNERR_INST_ERR, "invalid result shape!");
    }
    
    std::vector<float> scores;
    std::string result;
    
    std::vector<float> exps(seq_len*vocab_len, 0);
    int last_idx = 0;
    // TODO: move this search into model
    for(int s=0; s<seq_len; ++s) {
        double sum = 0.f;
        float max_score = -INFINITY;
        float max_score_pre_exp = -INFINITY;
        int max_idx = 0;
        for(int i=0; i<vocab_len; ++i) {
            float score = output_data[s * vocab_len + i];
            if (score > max_score_pre_exp) {
                max_score_pre_exp = score;
            }
        }
        for(int i=0; i<vocab_len; ++i) {
            float score = std::exp(output_data[s * vocab_len + i] - max_score_pre_exp);
            //output_data[s * vocab_len + i] = score;
            exps[s * vocab_len + i] = score;
            if (score > max_score) {
                max_score = score;
                max_idx = i;
            }
            sum += score;
        }

        if (max_idx > 0 && !(s > 0 && max_idx == last_idx)) {
            scores.emplace_back(max_score / static_cast<float>(sum));
            result.append(vocabulary_[max_idx - 1]);
        }
        last_idx = max_idx;
    }
    
    output->scores = scores;
    output->text = result;

    return status;
}

OCRTextRecognizer::~OCRTextRecognizer() {}

}

#endif // HAS_OPENCV
