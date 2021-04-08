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

#ifndef TNN_EXAMPLES_BASE_READING_COMPREHENSION_H_
#define TNN_EXAMPLES_BASE_READING_COMPREHENSION_H_

#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include "tnn_sdk_sample.h"

namespace TNN_NS {
    
class ReadingComprehensionInput : public TNNSDKInput {
public:
    ReadingComprehensionInput();
    void* inputIds;
    void* inputMask;
    void* segmentIds;
    virtual ~ReadingComprehensionInput();
};

class ReadingComprehensionOutput : public TNNSDKOutput {
public:
    ReadingComprehensionOutput();
    void *unstack0;
    void *unstack1;
    virtual ~ReadingComprehensionOutput();
};

class ReadingComprehension : public TNNSDKSample {
    ReadingComprehension();
    virtual ~ReadingComprehension();
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);
};

}
#endif