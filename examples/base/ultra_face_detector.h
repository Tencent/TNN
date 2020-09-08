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

#ifndef TNN_EXAMPLES_BASE_ULTRA_FACE_DETECTOR_H_
#define TNN_EXAMPLES_BASE_ULTRA_FACE_DETECTOR_H_ 

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "tnn_sdk_sample.h"

namespace TNN_NS {

struct FaceInfo : ObjectInfo {
} ;

class UltraFaceDetectorInput : public TNNSDKInput {
public:
    UltraFaceDetectorInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~UltraFaceDetectorInput();
};

class UltraFaceDetectorOutput : public TNNSDKOutput {
public:
    UltraFaceDetectorOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~UltraFaceDetectorOutput();
    std::vector<FaceInfo> face_list;
};

class UltraFaceDetectorOption : public TNNSDKOption {
public:
    UltraFaceDetectorOption();
    virtual ~UltraFaceDetectorOption();
    int input_width;
    int input_height;
    int num_thread = 1;
    float score_threshold = 0.7;
    float iou_threshold = 0.3;
    int topk = -1;
};

class UltraFaceDetector : public TNNSDKSample {
public:
    virtual ~UltraFaceDetector();
    
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);
    
private:
    void GenerateBBox(std::vector<FaceInfo> &bbox_collection, TNN_NS::Mat &scores, TNN_NS::Mat &boxes,
                      int image_w, int image_h,
                      float score_threshold, int num_anchors);

    void NMS(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, float iou_threshold, int type = 2);
    
private:
    int num_anchors;

    const float mean_vals[3] = {127, 127, 127};
    const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};

    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;

    std::vector<std::vector<float>> priors = {};
};

}

#endif // TNN_EXAMPLES_BASE_ULTRA_FACE_DETECTOR_H_
