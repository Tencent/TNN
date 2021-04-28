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

#ifndef TNN_EXAMPLES_BASE_OBJECT_DETECTOR_NANODET_H_
#define TNN_EXAMPLES_BASE_OBJECT_DETECTOR_NANODET_H_

#include "object_detector_yolo.h"
#include <memory>
#include <string>
#include <vector>

namespace TNN_NS {

using ObjectDetectorNanodetOutput = ObjectDetectorYoloOutput;

class ObjectDetectorNanodetOption : public TNNSDKOption {
public:
    ObjectDetectorNanodetOption() {}
    virtual ~ObjectDetectorNanodetOption() {}
    int input_width;
    int input_height;
    
    float score_threshold  = 0.4f;
    float iou_threshold    = 0.5f;
    std::string model_cfg  = "m";
};

typedef struct HeadInfo {
    std::string cls_output;
    std::string dis_output;
    int stride;
} HeadInfo;

class ObjectDetectorNanodet : public TNN_NS::TNNSDKSample {
public:
    ~ObjectDetectorNanodet();
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);
    
private:
    void DecodeDetectionResult(Mat *cls_mat, Mat *dis_mat, const int stride, std::vector<ObjectInfo>& detecs);
    void NMS(std::vector<ObjectInfo>& objs, std::vector<ObjectInfo>& results);
    
    float score_threshold = 0.4f;
    float iou_threshold   = 0.5f;
    int pads[4];

    const std::vector<HeadInfo> heads{
        {"cls_pred_stride_8",  "dis_pred_stride_8",  8},
        {"cls_pred_stride_16", "dis_pred_stride_16", 16},
        {"cls_pred_stride_32", "dis_pred_stride_32", 32},
    };
    const static int num_class = 80; // ms-coco detection
    int reg_max   = 10;
};

}

#endif // TNN_EXAMPLES_BASE_OBJECT_DETECTOR_NANODET_H_
