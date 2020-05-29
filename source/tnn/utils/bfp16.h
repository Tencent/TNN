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

#ifndef TNN_SOURCE_TNN_UTILS_BFP16_H_
#define TNN_SOURCE_TNN_UTILS_BFP16_H_

#include <stdint.h>

namespace TNN_NS {

typedef union {
    float f;
    uint32_t u;
} cvt_32b;

typedef struct bfp16_struct {
public:
    uint16_t w = 0;

    bfp16_struct() : w(0) {}

    bfp16_struct(float vf) {
        cvt_32b c;
        c.f = vf;
        w   = c.u >> 16;
    }

    operator const float() const {
        cvt_32b c;
        c.u = w << 16;
        return c.f;
    }
} bfp16_t;

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_UTILS_BFP16_H_
