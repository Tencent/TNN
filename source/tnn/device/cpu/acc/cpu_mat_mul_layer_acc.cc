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

#include "tnn/device/cpu/acc/cpu_unary_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
DECLARE_CPU_ACC(MatMul, LAYER_MATMUL);

Status CpuMatMulLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuMatMulLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param               = dynamic_cast<MatMulLayerParam *>(param_);
    auto resource            = dynamic_cast<MatMulLayerResource *>(resource_);
    DimsVector matrix_a_dims = param->matrix_a_dims;
    DimsVector matrix_b_dims = param->matrix_b_dims;
    if (matrix_a_dims.size() == 1) {
        matrix_a_dims.insert(matrix_a_dims.begin(), 1);
    }
    if (matrix_b_dims.size() == 1) {
        matrix_b_dims.push_back(1);
    }
    DataType data_type       = inputs[0]->GetBlobDesc().data_type;
    auto matrix_c_dims       = outputs[0]->GetBlobDesc().dims;
    if (data_type == DATA_TYPE_FLOAT) {
        float *matrix_a;
        float *matrix_b;

        if (inputs.size() == 2) {
            matrix_a = static_cast<float *>(inputs[0]->GetHandle().base);
            matrix_b = static_cast<float *>(inputs[1]->GetHandle().base);
        } else {
            auto weight = resource->weight.force_to<float *>();
            matrix_a    = param->weight_position == 0 ? weight : static_cast<float *>(inputs[0]->GetHandle().base);
            matrix_b    = param->weight_position == 1 ? weight : static_cast<float *>(inputs[0]->GetHandle().base);
        }
        auto matrix_c = static_cast<float *>(outputs[0]->GetHandle().base);
        int M         = matrix_a_dims[matrix_a_dims.size() - 2];
        int N         = matrix_a_dims[matrix_a_dims.size() - 1];
        int K         = matrix_b_dims[matrix_b_dims.size() - 1];
        int count     = DimsVectorUtils::Count(matrix_c_dims);
        int channel   = count / (M * K);
        for (int c = 0; c < channel; ++c) {
            for (int m = 0; m < M; ++m) {
                for (int k = 0; k < K; ++k) {
                    float sum = 0;
                    for (int n = 0; n < N; ++n) {
                        sum += matrix_a[c * M * N + m * N + n] * matrix_b[c * N * K + n * K + k];
                    }
                    matrix_c[c * M * K + m * K + k] = sum;
                }
            }
        }
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(MatMul, LAYER_MATMUL)

}  // namespace TNN_NS
