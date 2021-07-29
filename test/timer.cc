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

#include "test/timer.h"

#include <cmath>

namespace TNN_NS {

namespace test {

using std::chrono::duration_cast;
using std::chrono::microseconds;

Timer::Timer(std::string timer_info) {
    timer_info_ = timer_info;
    Reset();
}

void Timer::Start() {
    start_ = system_clock::now();
}

void Timer::Stop() {
    stop_ = system_clock::now();
    float delta = duration_cast<microseconds>(stop_ - start_).count() / 1000.0f;
    min_         = static_cast<float>(fmin(min_, delta));
    max_         = static_cast<float>(fmax(max_, delta));
    sum_ += delta;
    count_++;
}

void Timer::Reset() {
    min_ = FLT_MAX;
    max_ = FLT_MIN;
    sum_ = 0.0f;
    count_ = 0;
    stop_ = start_ = system_clock::now();
}
   
void Timer::Print() {
    char min_str[16];
    snprintf(min_str, 16, "%6.3f", min_);
    char max_str[16];
    snprintf(max_str, 16, "%6.3f", max_);
    char avg_str[16];
    snprintf(avg_str, 16, "%6.3f", sum_ / (float)count_);
    LOGI("%-45s TNN Benchmark time cost: min = %-8s ms  |  max = %-8s ms  |  avg = %-8s ms \n", timer_info_.c_str(),
         min_str, max_str, avg_str);
}

} // namespace test

} // namespace TNN_NS
