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

#ifndef TNN_EXAMPLES_UTILS_LANDMARK_SMOOTHING_FILTER_H_
#define TNN_EXAMPLES_UTILS_LANDMARK_SMOOTHING_FILTER_H_

#include "tnn/core/status.h"

#include "relative_velocity_filter.h"
#include "time_stamp.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <iterator>

namespace TNN_NS {

using NormalizedLandmark = std::tuple<float,float,float>;
using NormalizedLandmarkList = std::vector<NormalizedLandmark>;

using Normalized2DLandmark = std::pair<float,float>;
using Normalized2DLandmarkList = std::vector<Normalized2DLandmark>;

class VelocityFilter {
public:
    VelocityFilter(int window_size, float velocity_scale,
                   float min_allowed_object_scale, int target_fps)
    : target_fps_(target_fps),
    window_size_(window_size),
    velocity_scale_(velocity_scale),
    min_allowed_object_scale_(min_allowed_object_scale) {}
    
    TNN_NS::Status Reset();
    
    TNN_NS::Status Apply(const NormalizedLandmarkList& in_landmarks,
                         const std::pair<int, int>& image_size,
                         const TimeStamp& timestamp,
                         NormalizedLandmarkList* out_landmarks);
    
    TNN_NS::Status Apply2D(const Normalized2DLandmarkList& in_landmarks,
                           const std::pair<int, int>& image_size,
                           const TimeStamp& timestamp,
                           Normalized2DLandmarkList* out_landmarks);
    
private:
    // Initializes filters for the first time or after Reset. If initialized then
    // check the size.
    TNN_NS::Status InitializeFiltersIfEmpty(const size_t n_landmarks);
    // Check if a Landmark is valid
    static bool isValidLandMark(const NormalizedLandmark& m);
    static bool isValid2DLandMark(const Normalized2DLandmark& m);
    
    // desired fps
    int target_fps_;
    int window_size_;
    float velocity_scale_;
    float min_allowed_object_scale_;
    
    std::vector<TNN_NS::RelativeVelocityFilter> x_filters_;
    std::vector<TNN_NS::RelativeVelocityFilter> y_filters_;
    std::vector<TNN_NS::RelativeVelocityFilter> z_filters_;
};

}

#endif // TNN_EXAMPLES_UTILS_LANDMARK_SMOOTHING_FILTER_H_
