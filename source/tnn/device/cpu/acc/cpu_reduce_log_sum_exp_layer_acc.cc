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

#include "tnn/device/cpu/acc/cpu_reduce_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

DECLARE_CPU_PRE_REDUCE_POST_ACC(ReduceLogSumExp, LAYER_REDUCE_LOG_SUM_EXP);

Status CpuReduceLogSumExpLayerAcc::PreCalculateReduce(float* dst, float* src, int count) {
    ::memcpy(dst, src, count * sizeof(float));
    return TNN_OK;
}

Status CpuReduceLogSumExpLayerAcc::PreCalculateReduce(int32_t* dst, int32_t* src, int count) {
    ::memcpy(dst, src, count * sizeof(int32_t));
    return TNN_OK;
}

Status CpuReduceLogSumExpLayerAcc::CalculateReduce(float* output_data, float* input_data, int outer_dim, int channels,
                                                   int inner_dim) {
    for (int oc = 0; oc < outer_dim; oc++) {
        // Standardize to prevent overflow.
        // log(sum(exp xi)) = c + log(sum(exp(xi-c)))
        std::vector<float> max_values(inner_dim, -FLT_MAX);
        for (int c = 0; c < channels; c++) {
            const int offset = c * inner_dim;
            for (int ic = 0; ic < inner_dim; ic++) {
                max_values[ic] = std::max(max_values[ic], input_data[ic + offset]);
            }
        }

        for (int c = 0; c < channels; c++) {
            for (int ic = 0; ic < inner_dim; ic++) {
                output_data[ic] += std::exp(input_data[ic] - max_values[ic]);
            }
            input_data += inner_dim;
        }

        for (int ic = 0; ic < inner_dim; ic++) {
            output_data[ic] = std::log(output_data[ic]) + max_values[ic];
        }
        output_data += inner_dim;
    }
    return TNN_OK;
}

Status CpuReduceLogSumExpLayerAcc::CalculateReduce(int32_t* output_data, int32_t* input_data, int outer_dim, int channels,
                                                   int inner_dim) {
    for (int oc = 0; oc < outer_dim; oc++) {
        // Standardize to prevent overflow.
        // log(sum(exp xi)) = c + log(sum(exp(xi-c)))
        std::vector<int32_t> max_values(inner_dim, INT32_MIN);
        for (int c = 0; c < channels; c++) {
            const int offset = c * inner_dim;
            for (int ic = 0; ic < inner_dim; ic++) {
                max_values[ic] = std::max(max_values[ic], input_data[ic + offset]);
            }
        }

        for (int c = 0; c < channels; c++) {
            for (int ic = 0; ic < inner_dim; ic++) {
                output_data[ic] += std::exp(input_data[ic] - max_values[ic]);
            }
            input_data += inner_dim;
        }

        for (int ic = 0; ic < inner_dim; ic++) {
            output_data[ic] = std::log(output_data[ic]) + max_values[ic];
        }
        output_data += inner_dim;
    }
    return TNN_OK;
}

Status CpuReduceLogSumExpLayerAcc::PostCalculateReduce(float* dst, float* src, int count) {
    ::memcpy(dst, src, count * sizeof(float));
    return TNN_OK;
}

Status CpuReduceLogSumExpLayerAcc::PostCalculateReduce(int32_t* dst, int32_t* src, int count) {
    ::memcpy(dst, src, count * sizeof(int32_t));
    return TNN_OK;
}

REGISTER_CPU_REDUCE_ACC(ReduceLogSumExp, LAYER_REDUCE_LOG_SUM_EXP);

}  // namespace TNN_NS
