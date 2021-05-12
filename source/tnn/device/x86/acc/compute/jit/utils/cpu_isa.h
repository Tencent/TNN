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

#ifndef TNN_DEVICE_X86_ACC_COMPUTE_JIT_UTILS_CPU_ISA_HPP_
#define TNN_DEVICE_X86_ACC_COMPUTE_JIT_UTILS_CPU_ISA_HPP_


#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include <chrono>
#include <random>
#include <fstream>
#include <exception>

#include <immintrin.h>
#include <xmmintrin.h>

#include "tnn/device/x86/acc/compute/jit/common/type_def.h"

namespace TNN_NS {

typedef enum {
    sse42,
    avx,
    avx2,
    avx512,
    avx512_vnni,
} x86_isa_t;

bool cpu_with_isa(x86_isa_t arch);

} // namespace tnn

#endif // TNN_DEVICE_X86_ACC_COMPUTE_JIT_UTILS_CPU_ISA_HPP_
