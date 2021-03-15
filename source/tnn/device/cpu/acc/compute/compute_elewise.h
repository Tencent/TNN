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
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

/*
 * Output[i] = input0[i] op input1[i] op ... op  input..n[i]
 * CPU_ELEMENT_WISE supports broadcast on all dimensions
 */
template <typename T>
void CPU_ELEMENT_WISE(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
                      const DimsVector& shape_output, std::function<T(T, T)> op) {
    const int count = DimsVectorUtils::Count(shape_output);
    T *output_data  = static_cast<T *>(output);

    OMP_PARALLEL_FOR_
    for (int offset = 0; offset < count; ++offset) {
        DimsVector output_index = DimsOffsetUtils::ConvertOffsetToIndex(shape_output, offset);
        T result;
        for (int i = 0; i < input_ptrs.size(); i++) {
            T *input_data = static_cast<T *>(input_ptrs[i]);
            auto input_shape  = input_shapes[i];
            DimsVector input_index;
            auto diff = shape_output.size() - input_shape.size();
            for (int i = 0; i < input_shape.size(); ++i) {
                input_index.push_back(std::min(output_index[i + diff], input_shape[i] - 1));
            }
            int input_offset = DimsOffsetUtils::ConvertIndexToOffset(input_shape, input_index);
            if (i == 0) {
                result = input_data[input_offset];
            } else {
                result = op(result, input_data[input_offset]);
            }
        }
        output_data[offset] = result;
    }
}

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
