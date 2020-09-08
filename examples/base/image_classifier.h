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

#ifndef TNN_EXAMPLES_BASE_IMAGE_CLASSIFIER_H_
#define TNN_EXAMPLES_BASE_IMAGE_CLASSIFIER_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "tnn_sdk_sample.h"

namespace TNN_NS {

class ImageClassifierOutput : public TNNSDKOutput {
public:
    ImageClassifierOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~ImageClassifierOutput();
    
    int class_id = -1;
};

class ImageClassifier : public TNN_NS::TNNSDKSample {
public:
    virtual ~ImageClassifier();
    virtual MatConvertParam GetConvertParamForInput(std::string tag = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
};

}

#endif // TNN_EXAMPLES_BASE_IMAGE_CLASSIFIER_H_
