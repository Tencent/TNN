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

#ifndef TNN_CPU_COMPUTE_INT8_H_
#define TNN_CPU_COMPUTE_INT8_H_

#include <float.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>

#include "tnn/core/common.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

// int8 add, reload by scale
void CPU_ADD(const std::vector<void *> &input_ptrs, const std::vector<float *> &scale_ptrs, int scale_len, void *output,
             float *scale_out, DimsVector dims);

// int8 sub, reload by scale
void CPU_SUB(const std::vector<void *> &input_ptrs, const std::vector<float *> &scale_ptrs, int scale_len, void *output,
             float *scale_out, DimsVector dims);

// asy int8 add, reload by scale and bias
void CPU_ADD_BIAS(const std::vector<void *> &input_ptrs, const std::vector<float *> &scale_ptrs,
             const std::vector<int8_t *> &zero_point_ptrs, int scale_len, void *output, float *scale_out,
             int8_t *zero_point_out, DimsVector dims);

// asy int8 sub, reload by scale and bias
void CPU_SUB_BIAS(const std::vector<void *> &input_ptrs, const std::vector<float *> &scale_ptrs,
             const std::vector<int8_t *> &zero_point_ptrs, int scale_len, void *output, float *scale_out,
             int8_t *zero_point_out, DimsVector dims);

}  // namespace TNN_NS

#endif  // TNN_CPU_COMPUTE_INT8_H_
