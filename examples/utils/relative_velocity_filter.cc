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

#include "relative_velocity_filter.h"

#include <cmath>

using namespace std::chrono;

namespace TNN_NS {

float RelativeVelocityFilter::Apply(const TimeStamp& timestamp, float value_scale,
                                    float value) {
    const auto new_timestamp = timestamp;
    if (last_timestamp_ >= new_timestamp) {
        // Results are unpredictable in this case, so nothing to do but
        // return same value
        return value;
    }
    
    float alpha;
    if (isEmpty(last_timestamp_)) {
        alpha = 1.0;
    } else {
        if(!(distance_mode_ == DistanceEstimationMode::kLegacyTransition ||
             distance_mode_ == DistanceEstimationMode::kForceCurrentScale))
            return 0;
        
        const float distance = distance_mode_ == DistanceEstimationMode::kLegacyTransition
        ? value * value_scale - last_value_ * last_value_scale_    // Original.
        : value_scale * (value - last_value_);  // Translation invariant.
        
        const int64_t duration = duration_cast<nanoseconds>(new_timestamp - last_timestamp_).count();
        
        float cumulative_distance = distance;
        int64_t cumulative_duration = duration;
        
        // Define max cumulative duration assuming
        // 30 frames per second is a good frame rate, so assuming 30 values
        // per second or 1 / 30 of a second is a good duration per window element
        const int64_t kAssumedMaxDuration = 1000000000 / target_fps_;
        const int64_t max_cumulative_duration = (1 + window_.size()) * kAssumedMaxDuration;
        for (const auto& el : window_) {
            if (cumulative_duration + el.duration > max_cumulative_duration) {
                // This helps in cases when durations are large and outdated
                // window elements have bad impact on filtering results
                break;
            }
            cumulative_distance += el.distance;
            cumulative_duration += el.duration;
        }
        
        constexpr double kNanoSecondsToSecond = 1e-9;
        const float velocity = cumulative_distance / (cumulative_duration * kNanoSecondsToSecond);
        alpha = 1.0f - 1.0f / (1.0f + velocity_scale_ * std::abs(velocity));
        window_.push_front({distance, duration});
        if (window_.size() > max_window_size_) {
            window_.pop_back();
        }
    }
    
    last_value_ = value;
    last_value_scale_ = value_scale;
    last_timestamp_ = new_timestamp;
    
    return low_pass_filter_.ApplyWithAlpha(value, alpha);
}

}
