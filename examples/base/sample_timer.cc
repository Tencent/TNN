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

#include "sample_timer.h"

#include <cmath>

namespace TNN_NS {

using std::chrono::duration_cast;
using std::chrono::microseconds;

void SampleTimer::Start() {
    start_ = system_clock::now();
}

void SampleTimer::Stop() {
    stop_ = system_clock::now();
}

double SampleTimer::GetTime() {
    return duration_cast<microseconds>(stop_ - start_).count() / 1000.0f;
}

void SampleTimer::Reset() {
    stop_ = start_ = system_clock::now();
}

}
