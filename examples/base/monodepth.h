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

#ifndef TNN_EXAMPLES_BASE_HAIR_SEGMENTATION_H_
#define TNN_EXAMPLES_BASE_HAIR_SEGMENTATION_H_

#include <algorithm>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <array>

#include "tnn_sdk_sample.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

class MonoDepthInput : public TNNSDKInput {
public:
    MonoDepthInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~MonoDepthInput() {}
};

class MonoDepthOutput : public TNNSDKOutput {
public:
    MonoDepthOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~MonoDepthOutput() {};

    ImageInfo depth;
};

class MonoDepthOption : public TNNSDKOption {
public:
    MonoDepthOption() {}
    virtual ~MonoDepthOption() {}
};

class MonoDepth : public TNN_NS::TNNSDKSample {
public:
    virtual ~MonoDepth() {}
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat, std::string name = kTNNSDKDefaultName);

private:
    std::shared_ptr<Mat> GenerateDepthImage(std::shared_ptr<Mat> alpha);
    // the original input image shape
    DimsVector orig_dims;
    // the original input image
    std::shared_ptr<Mat> input_image;
    // the color used on hair
    RGBA hair_color_ = {0x00, 0x00, 0xbb, 0x55}; // blue
};

}

#endif // TNN_EXAMPLES_BASE_HAIR_SEGMENTATION_H_
