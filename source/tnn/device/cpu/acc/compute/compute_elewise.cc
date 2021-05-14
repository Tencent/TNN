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

#include "tnn/device/cpu/acc/compute/compute_elewise.h"

#include <cstring>
#include <functional>
#include <type_traits>

#include "math.h"

#include "tnn/core/macro.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

typedef std::function<float(float, float)> ELEWISE_OP;

/*
 * Output[i] = input0[i] op input1[i] op ... op  input..n[i]
 * CPU_ELEWISE supports broadcast on all dimensions
 */
void CPU_ELEWISE(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
                 DimsVector shape_output, ELEWISE_OP op) {
    const int count        = DimsVectorUtils::Count(shape_output);
    float *output_data     = static_cast<float *>(output);

    OMP_PARALLEL_FOR_
    for(int offset = 0; offset< count; ++offset) {
        DimsVector output_index = DimsOffsetUtils::ConvertOffsetToIndex(shape_output, offset);
        float result;
        for (int i = 0; i < input_ptrs.size(); i++) {
            float *input_data = static_cast<float *>(input_ptrs[i]);
            auto input_shape  = input_shapes[i];

            DimsVector input_index;
            int diff = shape_output.size() - input_shape.size();
            for(int i = 0; i < input_shape.size(); ++i) {
                input_index.push_back(std::min(output_index[i + diff], input_shape[i] - 1));
            }
             
            int input_offset = DimsOffsetUtils::ConvertIndexToOffset(input_shape, input_index);
            if(i == 0) {
                result = input_data[input_offset];
            } else {
                result = op(result, input_data[input_offset]);
            }
        }
        output_data[offset] = result;
    }
}

/*
 * Output[i] = min(input0[i], input1[i], input..n[i])
 * Broadcast is supported on n, c, h, w dims
 */
void CPU_MIN(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output) {
    ELEWISE_OP min_op = [](float a, float b) -> float { return std::min(a, b); };
    CPU_ELEWISE(input_ptrs, input_shapes, output, shape_output, min_op);
}

/*
 * Output[i] = max(input0[i], input1[i], input..n[i])
 * Broadcast is supported on each dimension of NCHW
 */
void CPU_MAX(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output) {
    ELEWISE_OP max_op = [](float a, float b) -> float { return std::max(a, b); };
    CPU_ELEWISE(input_ptrs, input_shapes, output, shape_output, max_op);
}

/*
 * Output[i] = input0[i] * input1[i] * ... *  input..n[i]
 * Broadcast is supported on all dimensions
 */
void CPU_MUL(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output) {
    ELEWISE_OP mul_op = [](float a, float b) -> float { return a * b; };
    CPU_ELEWISE(input_ptrs, input_shapes, output, shape_output, mul_op);
}

/*
 * Output[i] = input0[i] + input1[i] + ... +  input..n[i]
 * CPU_ADD supports broadcast on all dimensions
 */
void CPU_ADD(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output) {
    ELEWISE_OP add_op = [](float a, float b) -> float { return a + b; };
    CPU_ELEWISE(input_ptrs, input_shapes, output, shape_output, add_op);
}

/*
 * Output[i] = input0[i] / input1[i] / ... /  input..n[i]
 * CPU_DIV supports broadcast on all dimensions
 */
void CPU_DIV(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output) {
    ELEWISE_OP div_op = [](float a, float b) -> float { return a / b; };
    CPU_ELEWISE(input_ptrs, input_shapes, output, shape_output, div_op);
}

/*
 * Output[i] = input0[i] - input1[i] - ... -  input..n[i]
 * CPU_SUB supports broadcast on all dimensions
 */
void CPU_SUB(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output) {
    ELEWISE_OP sub_op = [](float a, float b) -> float { return a - b; };
    CPU_ELEWISE(input_ptrs, input_shapes, output, shape_output, sub_op);
}

/*
 * Output[i] = input0[i] - input1[i] - ... -  input..n[i]
 * CPU_SQUARED_DIFFERENCE supports broadcast on all dimensions
 */
void CPU_SQUARED_DIFFERENCE(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes,
                            void *output, DimsVector shape_output) {
    ELEWISE_OP squared_difference_op = [](float a, float b) -> float { return (a - b) * (a - b); };
    CPU_ELEWISE(input_ptrs, input_shapes, output, shape_output, squared_difference_op);
}

}  // namespace TNN_NS
