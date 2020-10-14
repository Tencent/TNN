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

#ifndef TNN_TEST_UNIT_TEST_COMMON_H_
#define TNN_TEST_UNIT_TEST_COMMON_H_

#include <chrono>
#include <random>

#include "tnn/core/macro.h"
#include "tnn/core/abstract_device.h"
#include "tnn/core/context.h"
#include "tnn/interpreter/layer_resource.h"

namespace TNN_NS {

template <typename T>
int InitRandom(T* host_data, size_t n, T range);
template <typename T>
int InitRandom(T* host_data, size_t n, T range_min, T range_max);
IntScaleResource* CreateIntScale(int channel);
void SetUpEnvironment(AbstractDevice** cpu, AbstractDevice** device, Context** cpu_context, Context** device_context);

}  // namespace TNN_NS

#endif  // TNN_TEST_UNIT_TEST_COMMON_H_
