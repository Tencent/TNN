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

#include "tnn/device/arm/acc/arm_mul_layer_acc.h"

#include "tnn/utils/dims_function_utils.h"
#include "tnn/utils/omp_utils.h"

#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

namespace TNN_NS {

Status ArmMulLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = ArmBinaryLayerAcc::Init(context, param, resource, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }

    op_type_ = ArmBinaryOpType::kMUL;

    return TNN_OK;
}

ArmMulLayerAcc::~ArmMulLayerAcc() {}

bool ArmMulLayerAcc::DataTypeSupported(DataType data_type) {
    if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_HALF || data_type == DATA_TYPE_BFP16 ||
        data_type == DATA_TYPE_INT32)
        return true;
    else
        return false;
}

Status ArmMulLayerAcc::ExecInt32(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto output = outputs[0];

    if (output->GetBlobDesc().data_type != DATA_TYPE_INT32) {
        LOGE("Error: layer acc dont support datatype: %d\n", output->GetBlobDesc().data_type);
        return TNNERR_LAYER_ERR;
    }

    auto dims = output->GetBlobDesc().dims;
    int count = DimsFunctionUtils::GetDim(dims, 0) * ROUND_UP(DimsFunctionUtils::GetDim(dims, 1), 4) *
                DimsVectorUtils::Count(dims, 2);

    if (inputs.size() == 1) {
        if (broadcast_.GetDataCount() != count) {
            LOGE("Error: mismatch input and broadcast shape\n");
            return Status(TNNERR_LAYER_ERR, "Unsupported int32 mul layer's input and broadcast shape");
        }
    } else if (inputs.size() == 2) {
        if (!(DimsVectorUtils::Equal(inputs[1]->GetBlobDesc().dims, dims))) {
            LOGE("Error: mismatch input and output shape\n");
            return Status(TNNERR_LAYER_ERR, "Unsupported int32 mul layer's inputs shape");
        }
    } else {
        return Status(TNNERR_UNSUPPORT_NET, "INPUT > 2 NOT IMPLEMENT FOR INT32");
    }

    if (!(DimsVectorUtils::Equal(inputs[0]->GetBlobDesc().dims, dims))) {
        LOGE("Error: mismatch input and output shape\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported int32 mul layer's inputs shape");
    }

    auto output_ptr = reinterpret_cast<int32_t *>(GetBlobHandlePtr(output->GetHandle()));
    auto input0_ptr = reinterpret_cast<int32_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    auto input1_ptr = (inputs.size() == 1) ? broadcast_.force_to<int32_t *>()
                                           : reinterpret_cast<int32_t *>(GetBlobHandlePtr(inputs[1]->GetHandle()));
    ;

    OMP_PARALLEL_FOR_
    for (int i = 0; i < count; i += 4) {
#ifdef TNN_USE_NEON
        int32x4_t res = vmulq_s32(vld1q_s32(input0_ptr + i), vld1q_s32(input1_ptr + i));
        vst1q_s32(output_ptr + i, res);
#else
        for (int j = 0; j < 4; ++j) {
            output_ptr[i + j] = input0_ptr[i + j] * input1_ptr[i + j];
        }
#endif
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Mul, LAYER_MUL)
REGISTER_ARM_PRECISION_FP16(LAYER_MUL)
REGISTER_ARM_LAYOUT(LAYER_MUL, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
