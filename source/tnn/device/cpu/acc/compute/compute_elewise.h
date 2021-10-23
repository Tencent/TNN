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
 * Output[i] = input0[i] op input1[i]
 * CPU_ELEMENT_WISE supports broadcast on all dimensions
 */
template <typename T_IN, typename T_OUT>
void CPU_ELEMENT_WISE_COMPARE(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
                      const DimsVector& shape_output, std::function<T_OUT(T_IN, T_IN)> op) {
    const int count = DimsVectorUtils::Count(shape_output);
    T_OUT *output_data  = static_cast<T_OUT *>(output);
    ASSERT(input_ptrs.size() == 2);

    OMP_PARALLEL_FOR_
    for (int offset = 0; offset < count; ++offset) {
        DimsVector output_index = DimsOffsetUtils::ConvertOffsetToIndex(shape_output, offset);
        T_OUT result;
        T_IN inputs[2];
        for (int i = 0; i < input_ptrs.size(); i++) {
            T_IN *input_data = static_cast<T_IN *>(input_ptrs[i]);
            auto input_shape  = input_shapes[i];
            DimsVector input_index;
            auto diff = shape_output.size() - input_shape.size();
            for (int i = 0; i < input_shape.size(); ++i) {
                input_index.push_back(std::min(output_index[i + diff], input_shape[i] - 1));
            }
            int input_offset = DimsOffsetUtils::ConvertIndexToOffset(input_shape, input_index);
            inputs[i] = input_data[input_offset];
        }
        output_data[offset] = op(inputs[0], inputs[1]);
    }
}


/*
 * Output[i] = input0[i] op input1[i] op ... op  input..n[i]
 * CPU_ELEMENT_WISE supports broadcast on all dimensions
 */
template <typename T_IN, typename T_OUT>
void CPU_ELEMENT_WISE(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
                      const DimsVector& shape_output, std::function<T_OUT(T_IN, T_IN)> op) {
    const int count = DimsVectorUtils::Count(shape_output);
    T_OUT *output_data  = static_cast<T_OUT *>(output);

    OMP_PARALLEL_FOR_
    for (int offset = 0; offset < count; ++offset) {
        DimsVector output_index = DimsOffsetUtils::ConvertOffsetToIndex(shape_output, offset);
        T_OUT result;
        for (int i = 0; i < input_ptrs.size(); i++) {
            T_IN *input_data = static_cast<T_IN *>(input_ptrs[i]);
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

template <typename T_IN_0, typename T_IN_1, typename T_IN_2, typename T_OUT>
void CPU_ELEMENT_WISE(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
                      const DimsVector& shape_output, std::function<T_OUT(T_IN_0, T_IN_1, T_IN_2)> op) {
    const int count = DimsVectorUtils::Count(shape_output);
    T_OUT *output_data  = static_cast<T_OUT *>(output);

    OMP_PARALLEL_FOR_
    for (int offset = 0; offset < count; ++offset) {
        DimsVector output_index = DimsOffsetUtils::ConvertOffsetToIndex(shape_output, offset);
        
        T_IN_0 *input_data_0 = static_cast<T_IN_0 *>(input_ptrs[0]);
        T_IN_1 *input_data_1 = static_cast<T_IN_1 *>(input_ptrs[1]);
        T_IN_2 *input_data_2 = static_cast<T_IN_2 *>(input_ptrs[2]);
        T_OUT result;
        
        int input_offset[3] = {0,0,0};
        
        for (int ii=0; ii<3; ii++) {
            auto input_shape  = input_shapes[ii];
            DimsVector input_index;
            auto diff = shape_output.size() - input_shape.size();
            for (int i = 0; i < input_shape.size(); ++i) {
                input_index.push_back(std::min(output_index[i + diff], input_shape[i] - 1));
            }
            input_offset[ii] = DimsOffsetUtils::ConvertIndexToOffset(input_shape, input_index);
        }

        output_data[offset] = op(input_data_0[input_offset[0]], input_data_1[input_offset[1]], input_data_2[input_offset[2]]);
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
