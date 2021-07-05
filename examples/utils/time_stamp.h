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

#ifndef TNN_EXAMPLES_UTILS_TIME_STAMP_H_
#define TNN_EXAMPLES_UTILS_TIME_STAMP_H_

#include "tnn/core/macro.h"
#include <chrono>

namespace TNN_NS {

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> TimeStamp;

// get a timestamp reflecting current moment
TimeStamp Now();

// check if time_stamp is empty(i.e., not a valid time stamp)
bool isEmpty(const TimeStamp& time_stamp);

}

#endif // TNN_EXAMPLES_UTILS_TIME_STAMP_H_
