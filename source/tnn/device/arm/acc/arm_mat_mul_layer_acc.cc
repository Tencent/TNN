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

#include "tnn/device/arm/acc/arm_mat_mul_layer_acc.h"

#include "tnn/device/arm/acc/compute/compute.h"
#include "tnn/device/arm/acc/compute/gemm_function.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"
#if TNN_ARM82
#include "tnn/device/arm/acc/compute_arm82/compute_half.h"
#endif

namespace TNN_NS {

ArmMatMulLayerAcc::~ArmMatMulLayerAcc() {}

Status ArmMatMulLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    auto res = dynamic_cast<MatMulLayerResource *>(resource);

    if (!res) {
        if (inputs.size() == 2) {
            // weights are get from inputs
            return TNN_OK;
        } else {
            LOGE("ArmMatMulLayerAcc::Init resource is null\n");
            return Status(TNNERR_PARAM_ERR, "ArmMatMulLayerAcc::Init resource is null");
        }
    }

#if TNN_ARM82
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        RawBuffer weight_handle = res->weight;
        CHECK_PARAM_NULL(weight_handle.force_to<void *>());
        if (weight_handle.GetDataType() == DATA_TYPE_FLOAT) {
            buffer_weight_ = RawBuffer(weight_handle.GetDataCount() * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
            ConvertFromFloatToHalf(weight_handle.force_to<float *>(), buffer_weight_.force_to<fp16_t *>(),
                                   weight_handle.GetDataCount());
            buffer_weight_.SetDataType(DATA_TYPE_HALF);
        }
    }
#endif
    return TNN_OK;
}

template <typename T>
Status ArmMatMulLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
    DataType data_type = inputs[0]->GetBlobDesc().data_type;
    auto matrix_c_dims = outputs[0]->GetBlobDesc().dims;

    T *matrix_a;
    T *matrix_b;

    if (inputs.size() == 2) {
        matrix_a = reinterpret_cast<T *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
        matrix_b = reinterpret_cast<T *>(GetBlobHandlePtr(inputs[1]->GetHandle()));
    } else {
        auto weight = resource->weight.force_to<T *>();
        if (buffer_weight_.force_to<T *>()) {
            weight = buffer_weight_.force_to<T *>();
        }
        matrix_a =
            param->weight_position == 0 ? weight : reinterpret_cast<T *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
        matrix_b =
            param->weight_position == 1 ? weight : reinterpret_cast<T *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    }
    auto matrix_c = reinterpret_cast<T *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

    int N = matrix_b_dims[matrix_b_dims.size() - 1];
    int K = matrix_a_dims[matrix_a_dims.size() - 1];
    int M = matrix_a_dims[matrix_a_dims.size() - 2];

    auto data_byte_size = DataTypeUtils::GetBytesSize(data_type);
    size_t pack_a_size  = M * K * data_byte_size + NEON_KERNEL_EXTRA_LOAD;
    int n_pack          = 8;
    if (data_type == DATA_TYPE_HALF) {
        n_pack = 16;
    }
    size_t pack_b_size    = K * ROUND_UP(N, n_pack) * data_byte_size + NEON_KERNEL_EXTRA_LOAD;
    size_t workspace_size = pack_a_size + pack_b_size;
    char *workspace       = reinterpret_cast<char *>(context_->GetSharedWorkSpace(workspace_size));
    T *pack_a_ptr         = reinterpret_cast<T *>(workspace);
    T *pack_b_ptr         = reinterpret_cast<T *>(workspace + pack_a_size);

    int count_a = DimsVectorUtils::Count(matrix_a_dims);
    int count_b = DimsVectorUtils::Count(matrix_b_dims);
    int count_c = DimsVectorUtils::Count(matrix_c_dims);
    int batch_a = count_a / (M * K);
    int batch_b = count_b / (K * N);
    int batch_c = count_c / (M * N);

    for (int bc = 0; bc < batch_c; ++bc) {
        int ba     = bc < batch_a ? bc : 0;
        int bb     = bc < batch_b ? bc : 0;
        auto a_ptr = matrix_a + ba * M * K;
        auto b_ptr = matrix_b + bb * K * N;
        auto c_ptr = matrix_c + bc * M * N;

        memset(c_ptr, 0, M * N * data_byte_size);
        // row major A[M * K] * B[K * N] = C[M * n]
        GemmFloatPackAB(M, N, K, a_ptr, pack_a_ptr, K, b_ptr, pack_b_ptr, N, c_ptr, N);
    }

    return TNN_OK;
}

Status ArmMatMulLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_data_type = inputs[0]->GetBlobDesc().data_type;
    if (input_data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    }
#if TNN_ARM82
    else if (input_data_type == DATA_TYPE_HALF) {
        return Exec<fp16_t>(inputs, outputs);
    }
#endif  // TNN_ARM82
    else {
        LOGE("ARM MatMul not support data type: %d\n", input_data_type);
        return Status(TNNERR_LAYER_ERR, "ARM MatMul not support data type");
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(MatMul, LAYER_MATMUL);
REGISTER_ARM_PRECISION_FP16(LAYER_MATMUL)
REGISTER_ARM_LAYOUT(LAYER_MATMUL, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
