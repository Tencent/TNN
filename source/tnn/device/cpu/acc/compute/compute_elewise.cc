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

#include <vector>
#include <cstring>
#include <functional>
#include <type_traits>

#include "math.h"

#include "tnn/core/macro.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/dims_offset_utils.h"
#include "tnn/utils/omp_utils.h"


namespace TNN_NS {

std::vector<int> dims_to_steps(std::vector<int> dims) {
    std::vector<int> ret(dims.size(), 1);
    int cnt = 1;
    for(int i=dims.size() - 1;i>=0;i--) {
        if (dims[i] == 1) {
            ret[i] = 0;
        } else {
            ret[i] = cnt;
            cnt *= dims[i];
        }
    }
    return ret;
}

enum BINARY_OP_TYPE
{
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    MAX = 4,
    MIN = 5,
    SQUARED_DIFFERENCE = 6
};

template<BINARY_OP_TYPE type>
float binary_op(const float a, const float b) {
    return a;
}

template<> float binary_op<BINARY_OP_TYPE::ADD>(const float a, const float b) {
    return a + b;
}

template<> float binary_op<BINARY_OP_TYPE::SUB>(const float a, const float b) {
    return a - b;
}

template<> float binary_op<BINARY_OP_TYPE::MUL>(const float a, const float b) {
    return a * b;
}

template<> float binary_op<BINARY_OP_TYPE::DIV>(const float a, const float b) {
    return a / b;
}

template<> float binary_op<BINARY_OP_TYPE::MIN>(const float a, const float b) {
    return a < b ? a : b;
}

template<> float binary_op<BINARY_OP_TYPE::MAX>(const float a, const float b) {
    return a > b ? a : b;
}

template<> float binary_op<BINARY_OP_TYPE::SQUARED_DIFFERENCE>(const float a, const float b) {
    return (a - b) * (a - b);
}


template<BINARY_OP_TYPE type>
void binary_kernel(std::vector<int> output_dims, const float * a, std::vector<int> steps_a, const float * b, std::vector<int> steps_b, 
                    float * c, std::vector<int> steps_c) 
{

    size_t idx_a = 0;
    size_t idx_b = 0;
    size_t idx_c = 0;

    const int MAX_DIM = 5;

    int d[MAX_DIM] = {1, 1, 1 ,1 ,1};
    int step_a[MAX_DIM] = {0};
    int step_b[MAX_DIM] = {0};
    int step_c[MAX_DIM] = {0};

    int offset = MAX_DIM - output_dims.size();
    memcpy(d + offset, &output_dims[0], output_dims.size() * sizeof(float));
    memcpy(step_a + offset, &steps_a[0], steps_a.size() * sizeof(float));
    memcpy(step_b + offset, &steps_b[0], steps_b.size() * sizeof(float));
    memcpy(step_c + offset, &steps_c[0], steps_c.size() * sizeof(float));

    for(int d0=0;d0<d[0];d0++) {
        for(int d1=0;d1<d[1];d1++) {
            for(int d2=0;d2<d[2];d2++) {
                for(int d3=0;d3<d[3];d3++) {
                    for(int d4=0;d4<d[4];d4++) {
                        c[idx_c] = binary_op<type>(a[idx_a], b[idx_b]);
                        idx_a += step_a[4];
                        idx_b += step_b[4];
                        idx_c += step_c[4];
                    }
                    idx_a += (step_a[3] - d[4] * step_a[4]);
                    idx_b += (step_b[3] - d[4] * step_b[4]);
                    idx_c += (step_c[3] - d[4] * step_c[4]);
                }
                idx_a += (step_a[2] - d[3] * step_a[3]);
                idx_b += (step_b[2] - d[3] * step_b[3]);
                idx_c += (step_c[2] - d[3] * step_c[3]);
            }
            idx_a += (step_a[1] - d[2] * step_a[2]);
            idx_b += (step_b[1] - d[2] * step_b[2]);
            idx_c += (step_c[1] - d[2] * step_c[2]);
        }
        idx_a += (step_a[0] - d[1] * step_a[1]);
        idx_b += (step_b[0] - d[1] * step_b[1]);
        idx_c += (step_c[0] - d[1] * step_c[1]);
    }


}


/*
 * Output[i] = input0[i] op input1[i] op ... op  input..n[i]
 * CPU_ELEWISE supports broadcast on all dimensions
 */
template<BINARY_OP_TYPE type>
void CPU_ELEWISE(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
                 DimsVector shape_output) {

    const int count        = DimsVectorUtils::Count(shape_output);
    float *output_data     = static_cast<float *>(output);

    if (input_shapes[0].size() != input_shapes[1].size()) {
        LOGE("Error, shape len not equal\n");
        return;
    }

    std::vector<int> steps_a = dims_to_steps(input_shapes[0]);
    std::vector<int> steps_b = dims_to_steps(input_shapes[1]);
    std::vector<int> steps_c = dims_to_steps(shape_output);

    binary_kernel<type>(shape_output, (const float *)input_ptrs[0], steps_a, (const float *)input_ptrs[1], steps_b, (float *)output, steps_c);

}

/*
 * Output[i] = min(input0[i], input1[i], input..n[i])
 * Broadcast is supported on n, c, h, w dims
 */
void CPU_MIN(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output) {
    CPU_ELEWISE<BINARY_OP_TYPE::MIN>(input_ptrs, input_shapes, output, shape_output);
}

/*
 * Output[i] = max(input0[i], input1[i], input..n[i])
 * Broadcast is supported on each dimension of NCHW
 */
void CPU_MAX(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output) {
    CPU_ELEWISE<BINARY_OP_TYPE::MAX>(input_ptrs, input_shapes, output, shape_output);
}

/*
 * Output[i] = input0[i] * input1[i] * ... *  input..n[i]
 * Broadcast is supported on all dimensions
 */
void CPU_MUL(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output) {
    CPU_ELEWISE<BINARY_OP_TYPE::MUL>(input_ptrs, input_shapes, output, shape_output);
}

/*
 * Output[i] = input0[i] + input1[i] + ... +  input..n[i]
 * CPU_ADD supports broadcast on all dimensions
 */
void CPU_ADD(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output) {
    CPU_ELEWISE<BINARY_OP_TYPE::ADD>(input_ptrs, input_shapes, output, shape_output);
}

/*
 * Output[i] = input0[i] / input1[i] / ... /  input..n[i]
 * CPU_DIV supports broadcast on all dimensions
 */
void CPU_DIV(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output) {
    CPU_ELEWISE<BINARY_OP_TYPE::DIV>(input_ptrs, input_shapes, output, shape_output);
}

/*
 * Output[i] = input0[i] - input1[i] - ... -  input..n[i]
 * CPU_SUB supports broadcast on all dimensions
 */
void CPU_SUB(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, void *output,
             DimsVector shape_output) {
    CPU_ELEWISE<BINARY_OP_TYPE::SUB>(input_ptrs, input_shapes, output, shape_output);
}

/*
 * Output[i] = input0[i] - input1[i] - ... -  input..n[i]
 * CPU_SQUARED_DIFFERENCE supports broadcast on all dimensions
 */
void CPU_SQUARED_DIFFERENCE(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes,
                            void *output, DimsVector shape_output) {
    CPU_ELEWISE<BINARY_OP_TYPE::SQUARED_DIFFERENCE>(input_ptrs, input_shapes, output, shape_output);
}

}  // namespace TNN_NS
