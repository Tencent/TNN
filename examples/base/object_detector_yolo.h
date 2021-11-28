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

#ifndef TNN_EXAMPLES_BASE_OBJECT_DETECTOR_YOLO_H_
#define TNN_EXAMPLES_BASE_OBJECT_DETECTOR_YOLO_H_

#include "tnn_sdk_sample.h"
#include "detector_utils.h"
#include <memory>
#include <string>
#include <vector>

namespace TNN_NS {

class ObjectDetectorYoloOutput : public TNNSDKOutput {
public:
    ObjectDetectorYoloOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~ObjectDetectorYoloOutput();
    std::vector<ObjectInfo> object_list;
};

class ObjectDetectorYolo : public TNN_NS::TNNSDKSample {
public:
    ~ObjectDetectorYolo();
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);
    
private:
    void GenerateDetectResult(std::vector<std::shared_ptr<Mat> >outputs, std::vector<ObjectInfo>& detects,
                              int image_width, int image_height);
    void NMS(std::vector<ObjectInfo>& objs, std::vector<ObjectInfo>& results);

    void PostProcessMat(std::vector<std::shared_ptr<Mat> >outputs, std::vector<std::shared_ptr<Mat> >& post_mats);
    
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

}
#endif // TNN_EXAMPLES_BASE_OBJECT_DETECTOR_YOLO_H_
