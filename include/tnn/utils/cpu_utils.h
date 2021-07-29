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

#ifndef TNN_INCLUDE_TNN_UTILS_CPU_UTILS_H_
#define TNN_INCLUDE_TNN_UTILS_CPU_UTILS_H_

#include <utility>
#include <vector>

#include "tnn/core/macro.h"
#include "tnn/core/status.h"

namespace TNN_NS {

class CpuUtils {
public:
    // @brief set cpu affinity
    // @param cpu_list vector of cpuids("0,1,2,3")
    PUBLIC static Status SetCpuAffinity(const std::vector<int>& cpu_list);

    // @brief set cpu powersave
    // @param powersave 0:all cpus 1:little cluster 2:big cluster
    PUBLIC static Status SetCpuPowersave(int powersave);

    // @brief get cpu fp16 capability
    PUBLIC static bool CpuSupportFp16();

    // @brief get cpu int8 dot capability
    PUBLIC static bool CpuSupportInt8Dot();

    // @brief set x86 cpu denormal ftz and daz, no use for other cpu.
    // @param denormal 0:turn off denormal 1:turn on denormal
    PUBLIC static void SetCpuDenormal(int denormal);
};

}  // namespace TNN_NS

#endif  // TNN_INCLUDE_TNN_UTILS_CPU_UTILS_H_
