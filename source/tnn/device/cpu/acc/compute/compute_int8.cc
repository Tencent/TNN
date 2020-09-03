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
#include "tnn/utils/dims_vector_utils.h"
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

void CPU_DEQUANT(const int8_t *input_ptr, const float *scale_ptr, int scale_len, float *output, DimsVector dims) {
    int batch   = dims[0];
    int channel = dims[1];
    int count   = DimsVectorUtils::Count(dims, 2, 4);
    for (int n = 0; n < batch; n++) {
        OMP_PARALLEL_FOR_
        for (int c = 0; c < channel; c++) {
            int offset    = n * channel * count + c * count;
            int scale_idx = scale_len == 1 ? 0 : c;
            for (int hw = 0; hw < dims[2] * dims[3]; hw++) {
                output[offset + hw] = scale_ptr[scale_idx] * static_cast<float>(input_ptr[offset + hw]);
            }
        }
    }
}

void CPU_QUANT(const float *input_ptr, const float *scale_ptr, int scale_len, int8_t *output, DimsVector dims) {
    for (int n = 0; n < dims[0]; n++) {
        OMP_PARALLEL_FOR_
        for (int c = 0; c < dims[1]; c++) {
            int offset    = n * dims[1] * dims[2] * dims[3] + c * dims[2] * dims[3];
            int scale_idx = scale_len == 1 ? 0 : c;
            for (int hw = 0; hw < dims[2] * dims[3]; hw++) {
                if (scale_ptr[scale_idx] != 0)
                    output[offset + hw] = float2int8(input_ptr[offset + hw] / scale_ptr[scale_idx]);
                else
                    output[offset + hw] = 0;
            }
        }
    }
}

}  // namespace TNN_NS