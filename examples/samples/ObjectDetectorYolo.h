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

#ifndef ObjectDetectorYolo_h
#define ObjectDetectorYolo_h

#include "TNNSDKSample.h"
#include <memory>
#include <string>
#include <vector>


typedef struct ObjInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int classid;
} ObjInfo;

constexpr const char* coco_classes[] = {
"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
"hair drier", "toothbrush"};

class ObjectDetectorYolo : public TNN_NS::TNNSDKSample {
public:
    ~ObjectDetectorYolo();
    ObjectDetectorYolo(int input_width, int input_length, int num_thread_ = 4);
    
    int Detect(std::shared_ptr<TNN_NS::Mat> image, int image_height, int image_width, std::vector<ObjInfo>& obj_list);
private:
    void GenerateDetectResult(std::vector<ObjInfo>& detects);
    void NMS(std::vector<ObjInfo>& objs, std::vector<ObjInfo>& results);
    void Sigmoid(float* v, const unsigned int count);
    
    int num_thread;
    int in_w;
    int in_h;
    
    std::vector<std::shared_ptr<TNN_NS::Mat>> outputs_;
    std::vector<std::string> output_blob_names_ = {"428", "427", "426"};
    
    float conf_thres = 0.4;
    float iou_thres = 0.5;
    // yolov5s model configurations
    std::vector<float> strides_ = {32.f, 16.f, 8.f};
    std::vector<float> anchor_grids_ = {116.f, 90.f,  \
                                        156.f, 198.f, \
                                        373.f, 326.f, \
                                        30.f, 61.f,   \
                                        62.f, 45.f,   \
                                        59.f, 119.f,  \
                                        10.f, 13.f,   \
                                        16.f, 30.f,   \
                                        33.f, 23.f };
    float iou_threshold_ = 0.5f;
    int num_anchor_ = 3;
    int detect_dim_ = 85;
    int grid_per_input_ = 6;
    
};

#endif /* ObjectDetectorYolo_h */
