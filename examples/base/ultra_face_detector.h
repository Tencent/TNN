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

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;

    float *landmarks = nullptr;
} FaceInfo;

std::vector<FaceInfo> AdjustFaceInfoToOriginalSize(std::vector<FaceInfo> face_info,
                                                   int detect_image_height, int detect_image_width,
                                                   int orig_image_height, int orig_image_width);

class UltraFaceDetector : public TNN_NS::TNNSDKSample {
public:
    ~UltraFaceDetector();
    UltraFaceDetector(int input_width, int input_length, int num_thread_ = 4, float score_threshold_ = 0.7, 
                    float iou_threshold_ = 0.3, int topk_ = -1);
    
    int Detect(std::shared_ptr<TNN_NS::Mat> image, int image_height, int image_width, std::vector<FaceInfo> &face_list);
    
private:
    void GenerateBBox(std::vector<FaceInfo> &bbox_collection, TNN_NS::Mat &scores, TNN_NS::Mat &boxes, 
                    float score_threshold, int num_anchors);

    void NMS(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type = blending_nms);
    
private:

    int num_thread;
    int image_w;
    int image_h;

    int in_w;
    int in_h;
    int num_anchors;

    int topk;
    float score_threshold;
    float iou_threshold;


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
    std::vector<int> w_h_list;

    std::vector<std::vector<float>> priors = {};
};

#endif // TNN_EXAMPLES_BASE_ULTRA_FACE_DETECTOR_H_
