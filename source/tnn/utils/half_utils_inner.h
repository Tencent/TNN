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

#ifndef TNN_INCLUDE_TNN_UTILS_HALF_UTILS_INNER_H_
#define TNN_INCLUDE_TNN_UTILS_HALF_UTILS_INNER_H_

#ifdef TNN_ARM82_A64

#include <cstdint>
typedef __fp16 fp16_t;

typedef union {
    uint16_t u;
    fp16_t f;
} cvt_16b;

static fp16_t cvt_half_from_raw_uint16(uint16_t u) {
    cvt_16b c;
    c.u = u;
    return c.f;
}

// Largest finite value.
#define HALF_MAX    cvt_half_from_raw_uint16(uint16_t(0x7BFF))
// Smallest positive normal value.
#define HALF_MIN    cvt_half_from_raw_uint16(uint16_t(0x0400))
// Smallest finite value.
#define HALF_LOWEST cvt_half_from_raw_uint16(uint16_t(0xFBFF))

#else // TNN_ARM82_A64

#include "tnn/utils/half.hpp"
typedef half_float::half fp16_t;
// Largest finite value.
#define HALF_MAX    std::numeric_limits<half_float::half>::max()
// Smallest positive normal value.
#define HALF_MIN    std::numeric_limits<half_float::half>::min()
// Smallest finite value.
#define HALF_LOWEST std::numeric_limits<half_float::half>::lowest()

#endif // TNN_ARM82_A64

#include "tnn/utils/half_utils.h"

#endif  // TNN_INCLUDE_TNN_UTILS_HALF_UTILS_INNER_H_
