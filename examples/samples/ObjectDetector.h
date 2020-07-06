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

#ifndef ObjectDetector_h
#define ObjectDetector_h

#include "TNNSDKSample.h"
#include <memory>
#include <string>

typedef struct ObjInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int classid;
} ObjInfo;

constexpr const char* voc_classes[] = {
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
};

class ObjectDetector : public TNN_NS::TNNSDKSample {
public:
    ~ObjectDetector();
    ObjectDetector(int input_width, int input_length, int num_thread_ = 4);
    
    int Detect(std::shared_ptr<TNN_NS::Mat> image, int image_height, int image_width, std::vector<ObjInfo>& obj_list);
private:
    void GenerateDetectResult(std::shared_ptr<TNN_NS::Mat> output, std::vector<ObjInfo>& detects);
    
    int num_thread;
    int in_w;
    int in_h;
    
    int num_detections_;
    std::string detection_output_name_ = "detection_out";
    
};

#endif /* ObjectDetector_h */
