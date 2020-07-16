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

#ifndef TNN_TEST_TIMER_H_
#define TNN_TEST_TIMER_H_

#include <chrono>
#include <string>

#include "tnn/core/macro.h"

namespace TNN_NS {

namespace test {

using std::chrono::time_point;
using std::chrono::system_clock;

class Timer {
public:
    Timer(std::string timer_info);
    void Start();
    void Stop();
    void Reset();
    void Print();

private:
    float min_;
    float max_;
    float sum_;
    std::string timer_info_;
    time_point<system_clock> start_;
    time_point<system_clock> stop_;
    int count_;
};

} // namespace test

} // namespace TNN_NS

#endif // TNN_TEST_TIMER_H_ 
