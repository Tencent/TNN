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

#include "tnn/device/arm/acc/arm_where_layer_acc.h"
#include "tnn/device/arm/acc/compute/binary_function.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

namespace TNN_NS {

bool ArmWhereLayerAcc::UseNaiveConstantBlobs() {
    return true;
}

ArmWhereLayerAcc::~ArmWhereLayerAcc() {}

template <bool reverse>
Status ArmWhereLayerAcc::Exec_Fp32(Blob *in_0, Blob *in_1, Blob *in_2, Blob *out) {
    float *input0_data  = reinterpret_cast<float*>(GetBlobHandlePtr(in_0->GetHandle()));
    float *input1_data  = reinterpret_cast<float*>(GetBlobHandlePtr(in_1->GetHandle()));
    int8_t *input2_data = reinterpret_cast<int8_t*>(GetBlobHandlePtr(in_2->GetHandle()));
    float *output_data  = reinterpret_cast<float*>(GetBlobHandlePtr(out->GetHandle()));

    if (!DimsVectorUtils::Equal(in_0->GetBlobDesc().dims, in_2->GetBlobDesc().dims)) {
        int8_t *broadcast = reinterpret_cast<int8_t*>(context_->GetSharedWorkSpace(DimsVectorUtils::Count(in_0->GetBlobDesc().dims) + NEON_KERNEL_EXTRA_LOAD));
        Broadcast<int8_t>(input2_data, in_2->GetBlobDesc().dims, broadcast, in_0->GetBlobDesc().dims);
        input2_data = broadcast;
    }

    auto dims         = out->GetBlobDesc().dims;
    auto count        = DimsVectorUtils::Count(dims);

#ifdef TNN_USE_NEON
    float32x4_t b    = vdupq_n_f32(input1_data[0]);
    int32x4_t v_zero = vdupq_n_s32(0);
#else
    float b = input1_data[0];
#endif

    OMP_PARALLEL_FOR_
    for (int i = 0; i < UP_DIV(count, 8); ++i) {
#ifdef TNN_USE_NEON
        int16x8_t c  = vmovl_s8(vld1_s8(input2_data + (i<<3)));
        int32x4_t cl = vmovl_s16(vget_low_s16(c));
        int32x4_t ch = vmovl_s16(vget_high_s16(c));
        uint32x4_t c0  = vceqq_s32(cl, v_zero);
        uint32x4_t c1  = vceqq_s32(ch, v_zero);

        float32x4_t a0 = vld1q_f32(input0_data + (i<<3));
        float32x4_t a1 = vld1q_f32(input0_data + (i<<3) + 4);
        if (reverse) {
            vst1q_f32(output_data + (i<<3),     vbslq_f32(c0, a0, b));
            vst1q_f32(output_data + (i<<3) + 4, vbslq_f32(c1, a1, b));
        } else {
            vst1q_f32(output_data + (i<<3),     vbslq_f32(c0, b, a0));
            vst1q_f32(output_data + (i<<3) + 4, vbslq_f32(c1, b, a1));
        }
#else
        for (int j = 0; j < 8; ++j) {
            if (reverse) {
                output_data[(i<<3)+j] = input2_data[(i<<3)+j] == 0 ? input0_data[(i<<3)+j] : b;
            } else {
                output_data[(i<<3)+j] = input2_data[(i<<3)+j] == 0 ? b : input0_data[(i<<3)+j];
            }
        }
#endif
    }

    return TNN_OK;
}

Status ArmWhereLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs.size() != 3) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "Where layer's inputs size must be 2");
    }

    auto input_dtype = inputs[0]->GetBlobDesc().data_type;

    if (input_dtype != inputs[1]->GetBlobDesc().data_type) {
        LOGE("Error: invalid inputs dtype\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported where layer's inputs dtype");
    }

    if (inputs[2]->GetBlobDesc().data_type != DATA_TYPE_INT8) {
        LOGE("Error: invalid inputs dtype\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported where layer's inputs dtype");
    }

    if (outputs[0]->GetBlobDesc().data_type != input_dtype) {
        LOGE("Error: invalid output dtype\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported where layer's output dtype");
    }

    Blob *in_0 = inputs[0];
    Blob *in_1 = inputs[1];
    Blob *out  = outputs[0];
    bool reverse = false;
    if (DimsVectorUtils::Count(in_0->GetBlobDesc().dims) == 1) {
        std::swap(in_0, in_1);
        reverse = true;
    }

    if (!(DimsVectorUtils::Count(in_1->GetBlobDesc().dims) == 1)) {
        LOGE("Error: invalid input shape\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported where layer's input shape");
    }

    if ((DimsVectorUtils::Count(in_0->GetBlobDesc().dims) != DimsVectorUtils::Count(out->GetBlobDesc().dims))) {
        LOGE("Error: mismatch input and output shape\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported where layer's input and output shape");
    }

    if (input_dtype == DATA_TYPE_FLOAT) {
        if (reverse) {
            return Exec_Fp32<true>(in_0, in_1, inputs[2], out);
        } else {
            return Exec_Fp32<false>(in_0, in_1, inputs[2], out);
        }
    }
// #if TNN_ARM82
//     else if (input_dtype == DATA_TYPE_HALF) {
//         Exec_Fp16(in_0, in_1, inputs[2], out);
//     }
// #endif
    else {
        LOGE("Error: unsupport input dtype\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported where layer's input dtype");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Where, LAYER_WHERE);
// REGISTER_ARM_PRECISION_FP16(LAYER_WHERE)
REGISTER_ARM_LAYOUT(LAYER_WHERE, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
