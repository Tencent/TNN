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

class HairSegmentationInput : public TNNSDKInput {
public:
    HairSegmentationInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~HairSegmentationInput() {}
};

class HairSegmentationOutput : public TNNSDKOutput {
public:
    HairSegmentationOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~HairSegmentationOutput() {};

    ImageInfo hair_mask;
    ImageInfo merged_image;
};

class HairSegmentationOption : public TNNSDKOption {
public:
    HairSegmentationOption() {}
    virtual ~HairSegmentationOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    // the processing mode of output mask
    int mode = 0;
};

class HairSegmentation : public TNN_NS::TNNSDKSample {
public:
    virtual ~HairSegmentation() {}
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat, std::string name = kTNNSDKDefaultName);
    // Set the color used on hair
    void SetHairColor(const RGBA& color) {
        this->hair_color_ = color;
    }

private:
    std::shared_ptr<Mat> ProcessAlpha(std::shared_ptr<Mat> alpha, int mode);
    std::shared_ptr<Mat> GenerateAlphaImage(std::shared_ptr<Mat> alpha);
    std::shared_ptr<Mat> MergeImage(std::shared_ptr<Mat> alpha, RGBA color);
    Status ResizeFloatMat(std::shared_ptr<Mat> input_mat, std::shared_ptr<Mat> output_mat, TNNInterpType type = TNNInterpLinear);
    Status ConvertMat(std::shared_ptr<Mat>src, std::shared_ptr<Mat>dst);
    // the original input image shape
    DimsVector orig_dims;
    // the original input image
    std::shared_ptr<Mat> input_image;
    // the color used on hair
    RGBA hair_color_ = {0x00, 0x00, 0xbb, 0x55}; // blue
};

}

#endif // TNN_EXAMPLES_BASE_HAIR_SEGMENTATION_H_
