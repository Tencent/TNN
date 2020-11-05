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

#ifndef TNN_EXAMPLES_BASE_POSE_DETECT_LANDMARK_H_
#define TNN_EXAMPLES_BASE_POSE_DETECT_LANDMARK_H_

#include <algorithm>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <array>

#include "tnn_sdk_sample.h"
#include "blazepose_detector.h"
#include "blazepose_landmark.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS{

class PoseDetectLandmark : public TNN_NS::TNNSDKComposeSample {
public:
    virtual ~PoseDetectLandmark() {}
    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);
    virtual Status Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks);
protected:
    std::shared_ptr<TNNSDKSample> predictor_detect_ = nullptr;
    std::shared_ptr<TNNSDKSample> predictor_landmark_ = nullptr;
private:
    DimsVector origin_input_shape;
    BlazePoseLandmark::RoIGenOptions detect2toi_option = {
        2,     // keypoints_start_idx
        3,     // keypoints_end_idx
        90.0f, // rotation_target_angle, in degree
        1.5f,  // scale_x
        1.5f   // scale_y
    };
};

}

#endif // TNN_EXAMPLES_BASE_POSE_DETECT_LANDMARK_H_

