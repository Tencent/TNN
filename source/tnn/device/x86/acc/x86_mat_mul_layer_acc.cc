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

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/device/x86/acc/x86_mat_mul_layer_acc.h"
#include "tnn/interpreter/layer_resource_generator.h"

namespace TNN_NS {

X86MatMulLayerAcc::~X86MatMulLayerAcc() {}

Status X86MatMulLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret;
    if (inputs.size() == 2) {
        ret = X86LayerAcc::Init(context, param, resource, inputs, outputs);
        RETURN_ON_NEQ(ret, TNN_OK);
        return TNN_OK;
    }

    auto res = dynamic_cast<MatMulLayerResource *>(resource);
    CHECK_PARAM_NULL(res);

    if (res->weight.GetDataType() == DATA_TYPE_HALF) {
        LayerResource *fp32_res = nullptr;
        RETURN_ON_NEQ(ConvertHalfResource(LAYER_MATMUL, res, &fp32_res), TNN_OK);
        matmul_acc_f32_resource_ = std::shared_ptr<LayerResource>(fp32_res);
        ret                      = X86LayerAcc::Init(context, param, matmul_acc_f32_resource_.get(), inputs, outputs);
    } else {
        ret = X86LayerAcc::Init(context, param, resource, inputs, outputs);
    }

    RETURN_ON_NEQ(ret, TNN_OK);
    return TNN_OK;
}

Status X86MatMulLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
            matrix_a = handle_ptr<float *>(inputs[0]->GetHandle());
            matrix_b = handle_ptr<float *>(inputs[1]->GetHandle());
        } else {
            auto weight = resource->weight.force_to<float *>();
            matrix_a    = param->weight_position == 0 ? weight : handle_ptr<float *>(inputs[0]->GetHandle());
            matrix_b    = param->weight_position == 1 ? weight : handle_ptr<float *>(inputs[0]->GetHandle());
        }
        auto matrix_c = handle_ptr<float *>(outputs[0]->GetHandle());

        int k_c = conv_gemm_conf_.K_c_;
        int m_c = conv_gemm_conf_.M_c_;
        int n_block = conv_gemm_conf_.n_block_;

        int M = matrix_b_dims[matrix_b_dims.size() - 1];
        int K = matrix_a_dims[matrix_a_dims.size() - 1];
        int N = matrix_a_dims[matrix_a_dims.size() - 2];

        size_t pack_a_size = ROUND_UP(m_c * k_c * sizeof(float), 32);
        size_t pack_b_size = k_c * ROUND_UP(N, n_block) * sizeof(float);
        size_t workspace_size = pack_a_size + pack_b_size;
        float *workspace = reinterpret_cast<float *>(context_->GetSharedWorkSpace(workspace_size));

        RawBuffer fake_bias(N * sizeof(float));
        float *fake_bias_ptr = fake_bias.force_to<float *>();

        int count_a     = DimsVectorUtils::Count(matrix_a_dims);
        int count_b     = DimsVectorUtils::Count(matrix_b_dims);
        int count_c     = DimsVectorUtils::Count(matrix_c_dims);
        int batch_a   = count_a / (K * N);
        int batch_b   = count_b / (M * K);
        int batch_c   = count_c / (M * N);
        for (int bc = 0; bc < batch_c; ++bc) {
            int ba = bc < batch_a ? bc : 0;
            int bb = bc < batch_b ? bc : 0;
            auto a_ptr = matrix_a + ba * K * N;
            auto b_ptr = matrix_b + bb * M * K;
            auto c_ptr = matrix_c + bc * M * N;

            // row major A[N * K] * B[K * M] = C[N * M]
            // equals to
            // col major B[M * K] * A[K * N] = C[M * N]
            conv_sgemm_nn_col_major(M, N, K, b_ptr, M, a_ptr, K, c_ptr, M,
                fake_bias_ptr, ActivationType_None, workspace, conv_gemm_conf_);
        }
    }

    return TNN_OK;
}

REGISTER_X86_ACC(MatMul, LAYER_MATMUL)

}  // namespace TNN_NS
