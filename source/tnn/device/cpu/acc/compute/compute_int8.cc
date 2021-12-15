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

#include "tnn/device/cpu/acc/compute/compute_int8.h"

#include "tnn/core/macro.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

// use float data type for intermediate result
typedef std::function<float(float, float)> INT8_OP;

void CPU_INT8_CALCULATE(const std::vector<void *> &input_ptrs, const std::vector<float *> &scale_ptrs, int scale_len,
                        void *output, float *scale_out, DimsVector dims, INT8_OP op) {
    int batch   = dims[0];
    int channel = dims[1];
    int count   = DimsVectorUtils::Count(dims, 2, 4);
    for (int n = 0; n < batch; n++) {
        OMP_PARALLEL_FOR_
        for (int c = 0; c < channel; c++) {
            int offset    = n * channel * count + c * count;
            int scale_idx = scale_len == 1 ? 0 : c;
            for (int hw = 0; hw < count; hw++) {
                float acc = 0;
                for (int inid = 0; inid < input_ptrs.size(); inid++) {
                    if (inid == 0) {
                        acc = scale_ptrs[inid][scale_idx] *
                              static_cast<float>(static_cast<int8_t *>(input_ptrs[inid])[hw + offset]);
                    } else {
                        acc = op(acc, scale_ptrs[inid][scale_idx] *
                                          static_cast<float>(static_cast<int8_t *>(input_ptrs[inid])[hw + offset]));
                    }
                }
                static_cast<int8_t *>(output)[hw + offset] = float2int8(acc / scale_out[scale_idx]);
            }
        }
    }
}
void CPU_INT8_BIAS_CALCULATE(const std::vector<void *> &input_ptrs, const std::vector<float *> &scale_ptrs,
                             const std::vector<int8_t *> &zero_point_ptrs, int scale_len, void *output,
                             float *scale_out, int8_t *zero_point_out, DimsVector dims, INT8_OP op) {
    int batch   = dims[0];
    int channel = dims[1];
    int count   = DimsVectorUtils::Count(dims, 2, 4);
    for (int n = 0; n < batch; n++) {
        OMP_PARALLEL_FOR_
        for (int c = 0; c < channel; c++) {
            int offset    = n * channel * count + c * count;
            int scale_idx = scale_len == 1 ? 0 : c;
            for (int hw = 0; hw < count; hw++) {
                float acc = 0;
                for (int inid = 0; inid < input_ptrs.size(); inid++) {
                    if (inid == 0) {
                        acc = scale_ptrs[inid][scale_idx] *
                              (static_cast<float>(static_cast<int8_t *>(input_ptrs[inid])[hw + offset]) -
                               static_cast<float>(static_cast<int8_t *>(zero_point_ptrs[inid])[scale_idx]));
                    } else {
                        acc = op(acc,
                                 scale_ptrs[inid][scale_idx] *
                                     (static_cast<float>(static_cast<int8_t *>(input_ptrs[inid])[hw + offset]) -
                                      static_cast<float>(static_cast<int8_t *>(zero_point_ptrs[inid])[scale_idx])));
                    }
                }
                static_cast<int8_t *>(output)[hw + offset] =
                    float2int8(acc / scale_out[scale_idx] + static_cast<float>(zero_point_out[scale_idx]));
            }
        }
    }
}

void CPU_ADD(const std::vector<void *> &input_ptrs, const std::vector<float *> &scale_ptrs, int scale_len, void *output,
             float *scale_out, DimsVector dims) {
    INT8_OP add_op = [](float a, float b) -> float { return a + b; };
    CPU_INT8_CALCULATE(input_ptrs, scale_ptrs, scale_len, output, scale_out, dims, add_op);
}
void CPU_SUB(const std::vector<void *> &input_ptrs, const std::vector<float *> &scale_ptrs, int scale_len, void *output,
             float *scale_out, DimsVector dims) {
    INT8_OP sub_op = [](float a, float b) -> float { return a - b; };
    CPU_INT8_CALCULATE(input_ptrs, scale_ptrs, scale_len, output, scale_out, dims, sub_op);
}
void CPU_ADD_BIAS(const std::vector<void *> &input_ptrs, const std::vector<float *> &scale_ptrs,
                  const std::vector<int8_t *> &zero_point_ptrs, int scale_len, void *output, float *scale_out,
                  int8_t *zero_point_out, DimsVector dims) {
    INT8_OP add_op = [](float a, float b) -> float { return a + b; };
    CPU_INT8_BIAS_CALCULATE(input_ptrs, scale_ptrs, zero_point_ptrs, scale_len, output, scale_out, zero_point_out, dims,
                            add_op);
}
void CPU_SUB_BIAS(const std::vector<void *> &input_ptrs, const std::vector<float *> &scale_ptrs,
                  const std::vector<int8_t *> &zero_point_ptrs, int scale_len, void *output, float *scale_out,
                  int8_t *zero_point_out, DimsVector dims) {
    INT8_OP sub_op = [](float a, float b) -> float { return a - b; };
    CPU_INT8_BIAS_CALCULATE(input_ptrs, scale_ptrs, zero_point_ptrs, scale_len, output, scale_out, zero_point_out, dims,
                            sub_op);
}
}  // namespace TNN_NS
