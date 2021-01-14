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

#include "low_pass_filter.h"

namespace TNN_NS {

LowPassFilter::LowPassFilter(float alpha) : initialized_{false} {
    SetAlpha(alpha);
}

float LowPassFilter::Apply(float value) {
    float result;
    if (initialized_) {
        result = alpha_ * value + (1.0 - alpha_) * stored_value_;
    } else {
        result = value;
        initialized_ = true;
    }
    raw_value_ = value;
    stored_value_ = result;
    return result;
}

float LowPassFilter::ApplyWithAlpha(float value, float alpha) {
    SetAlpha(alpha);
    return Apply(value);
}

bool LowPassFilter::HasLastRawValue() { return initialized_; }

float LowPassFilter::LastRawValue() { return raw_value_; }

float LowPassFilter::LastValue() { return stored_value_; }

void LowPassFilter::SetAlpha(float alpha) {
    if (alpha < 0.0f || alpha > 1.0f) {
        return;
    }
    alpha_ = alpha;
}

}
