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

#include "tnn/utils/naive_compute.h"
#include "tnn/device/cpu/acc/cpu_reduce_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_REDUCE_ACC(ReduceLogSumExp, LAYER_REDUCE_LOG_SUM_EXP);

Status CpuReduceLogSumExpLayerAcc::CalculateReduce(float* output_data, float* input_data, int outer_dim, int channels,
                                                   int inner_dim) {
    float* origin_output_data = output_data;
    int output_size           = outer_dim * inner_dim;
    for (int oc = 0; oc < outer_dim; oc++) {
        auto input_out = input_data + oc * channels * inner_dim;
        auto output_out = output_data + oc * inner_dim;
        for (int ic = 0; ic < inner_dim; ic++) {
            auto input_in = input_out + ic;
            auto output_in = output_out + ic;
            for (int c = 0; c < channels; c++) {
                *output_in += std::exp(input_in[c * inner_dim]);
            }
        }
    }

    for (int i = 0; i < output_size; ++i) {
        origin_output_data[i] = std::log(origin_output_data[i]);
    }
    return TNN_OK;
}

REGISTER_CPU_REDUCE_ACC(ReduceLogSumExp, LAYER_REDUCE_LOG_SUM_EXP);

}  // namespace TNN_NS
