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

#ifndef TNN_EXAMPLES_BASE_OBJECT_DETECTOR_SSD_H_
#define TNN_EXAMPLES_BASE_OBJECT_DETECTOR_SSD_H_

#include <memory>
#include <string>

#include "detector_utils.h"
#include "tnn_sdk_sample.h"

namespace TNN_NS {

//
class ObjectDetectorSSDOutput : public TNNSDKOutput {
public:
    ObjectDetectorSSDOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat){};
    virtual ~ObjectDetectorSSDOutput();
    std::vector<ObjectInfo> object_list;
};

class ObjectDetectorSSD : public TNNSDKSample {
public:
    ~ObjectDetectorSSD();
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat, std::string name = kTNNSDKDefaultName);

private:
    void GenerateObjects(std::vector<ObjectInfo>& objects, std::shared_ptr<Mat> scores, std::shared_ptr<Mat> boxes,
                         float score_threshold, int image_width, int image_height);
};

}  // namespace TNN_NS

#endif // TNN_EXAMPLES_BASE_OBJECT_DETECTOR_SSD_H_
