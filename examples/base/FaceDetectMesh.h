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

#ifndef FaceDetectMesh_hpp
#define FaceDetectMesh_hpp

#include <algorithm>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <array>

#include "TNNSDKSample.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS{

class FaceDetectMesh : public TNN_NS::TNNSDKComposeSample {
public:
    virtual ~FaceDetectMesh() {}

    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);

    virtual Status Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks);

protected:
    std::shared_ptr<TNNSDKSample> predictor_detect_ = nullptr;
    std::shared_ptr<TNNSDKSample> predictor_mesh_ = nullptr;
};

}

#endif /* FaceDetectMesh_hpp */
