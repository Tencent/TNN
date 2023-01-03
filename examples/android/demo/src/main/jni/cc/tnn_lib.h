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


#ifndef ANDROID_TNN_LIB_H_
#define ANDROID_TNN_LIB_H_

#include "tnn/core/tnn.h"
#include "tnn/core/instance.h"

#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110

class TNNLib {
public:
    TNNLib();

    int Init(const std::string& proto_file, const std::string& model_file, const std::string& device);

    std::vector<float> Forward(void* sourcePixelscolor);

    ~TNNLib();

private:

    TNN_NS::TNN tnn_;
    std::shared_ptr<TNN_NS::Instance> instance_;

};

#endif // ANDROID_TNN_LIB_H_
