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

#ifndef TNN_EXAMPLES_BASE_FACE_GRAY_TRANSFER_H_
#define TNN_EXAMPLES_BASE_FACE_GRAY_TRANSFER_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "tnn_sdk_sample.h"

namespace TNN_NS {

class FaceGrayTransfer : public TNN_NS::TNNSDKSample {
public:
    ~FaceGrayTransfer();
    virtual MatConvertParam GetConvertParamForInput(std::string tag = "");
//    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
//    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
//    
////    TNN_NS::Status Trasfer(std::shared_ptr<TNN_NS::Mat> input_mat,
////                   std::shared_ptr<TNN_NS::Mat>&output_mat,
////                   int input_width, int input_length);
};

}
#endif //TNN_EXAMPLES_BASE_FACE_GRAY_TRANSFER_H_
