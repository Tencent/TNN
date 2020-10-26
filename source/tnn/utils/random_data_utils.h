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

#ifndef TNN_UTILS_RANDOM_DATA_UTILS_H_
#define TNN_UTILS_RANDOM_DATA_UTILS_H_

#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "tnn/core/macro.h"

namespace TNN_NS {

template <typename T>
int InitRandom(T* host_data, size_t n, T range);
template <typename T>
int InitRandom(T* host_data, size_t n, T range_min, T range_max);

}  // namespace TNN_NS

#endif  // TNN_UTILS_RANDOM_DATA_UTILS_H_
