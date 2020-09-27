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

#ifndef TNN_CPU_COMPUTE_ELEWISE_H_
#define TNN_CPU_COMPUTE_ELEWISE_H_

#include <float.h>
#include <stdint.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>

#include "tnn/core/common.h"

namespace TNN_NS {

// float add
void CPU_MIN(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output);

void CPU_MAX(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output);

void CPU_MUL(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output);

void CPU_ADD(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output);

void CPU_DIV(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output);

void CPU_SUB(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output);

void CPU_SQUARED_DIFFERENCE(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes,
                            void *output, DimsVector shape_output);
}  // namespace TNN_NS
#endif  // TNN_CPU_COMPUTE_ELEWISE_H_
