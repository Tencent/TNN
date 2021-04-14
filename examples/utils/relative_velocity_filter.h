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

#ifndef TNN_EXAMPLES_UTILS_RELATIVE_VELOCITY_FILTER_H_
#define TNN_EXAMPLES_UTILS_RELATIVE_VELOCITY_FILTER_H_

#include "low_pass_filter.h"
#include "time_stamp.h"
#include <deque>
#include <cstdint>
#include <chrono>

namespace TNN_NS {

class RelativeVelocityFilter {
public:
    enum class DistanceEstimationMode {
        // When the value scale changes, uses a heuristic
        // that is not translation invariant (see the implementation for details).
        kLegacyTransition,
        // The current (i.e. last) value scale is always used for scale estimation.
        // When using this mode, the filter is translation invariant, i.e.
        //     Filter(Data + Offset) = Filter(Data) + Offset.
        kForceCurrentScale,
        
        kDefault = kLegacyTransition
    };
    
public:
    RelativeVelocityFilter(size_t window_size, float velocity_scale, int target_fps,
                           DistanceEstimationMode distance_mode)
    : max_window_size_{window_size},
    window_{window_size},
    velocity_scale_{velocity_scale},
    target_fps_{target_fps},
    distance_mode_{distance_mode} {}
    
    RelativeVelocityFilter(size_t window_size, float velocity_scale, int target_fps)
    : RelativeVelocityFilter{window_size, velocity_scale, target_fps,
        DistanceEstimationMode::kDefault} {}
    
    float Apply(const TimeStamp& timestamp, float value_scale, float value);
    
private:
    struct WindowElement {
        float distance;
        int64_t duration;
    };
    
    float last_value_ = 0.0;
    float last_value_scale_ = 1.0;
    TimeStamp last_timestamp_;
    
    size_t max_window_size_;
    int target_fps_ = 30;
    std::deque<WindowElement> window_;
    LowPassFilter low_pass_filter_{1.0f};
    float velocity_scale_;
    DistanceEstimationMode distance_mode_;
};

}

#endif // TNN_EXAMPLES_UTILS_RELATIVE_VELOCITY_FILTER_H_
