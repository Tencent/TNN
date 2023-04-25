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

#if TNN_ARM82

#include "tnn/device/arm/acc/arm_where_layer_acc.h"
#include "tnn/device/arm/acc/compute/binary_function.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

namespace TNN_NS {

template <bool reverse>
Status ArmWhereLayerAcc::exec_fp16_single(Blob *in_0, Blob *in_1, Blob *in_2, Blob *out) {
    fp16_t *input0_data  = reinterpret_cast<fp16_t*>(GetBlobHandlePtr(in_0->GetHandle()));
    fp16_t *input1_data  = reinterpret_cast<fp16_t*>(GetBlobHandlePtr(in_1->GetHandle()));
    int8_t *input2_data = reinterpret_cast<int8_t*>(GetBlobHandlePtr(in_2->GetHandle()));
    fp16_t *output_data  = reinterpret_cast<fp16_t*>(GetBlobHandlePtr(out->GetHandle()));

    if (!DimsVectorUtils::Equal(in_0->GetBlobDesc().dims, in_2->GetBlobDesc().dims)) {
        int8_t *broadcast = reinterpret_cast<int8_t*>(context_->GetSharedWorkSpace(DimsVectorUtils::Count(in_0->GetBlobDesc().dims) + NEON_KERNEL_EXTRA_LOAD));
        Broadcast<int8_t>(input2_data, in_2->GetBlobDesc().dims, broadcast, in_0->GetBlobDesc().dims);
        input2_data = broadcast;
    }

    auto dims         = out->GetBlobDesc().dims;
    auto count        = DimsVectorUtils::Count(dims);

#ifdef TNN_USE_NEON
    float16x8_t b    = vdupq_n_f16(input1_data[0]);
    int16x8_t v_zero = vdupq_n_s16(0);
#else
    fp16_t b = input1_data[0];
#endif

    OMP_PARALLEL_FOR_
    for (int i = 0; i < UP_DIV(count, 8); ++i) {
#ifdef TNN_USE_NEON
        int16x8_t c  = vmovl_s8(vld1_s8(input2_data + (i<<3)));
        uint16x8_t c0 = vceqq_s16(c, v_zero);
        float16x8_t a = vld1q_f16(input0_data + (i<<3));
        if (reverse) {
            vst1q_f16(output_data + (i<<3), vbslq_f16(c0, a, b));
        } else {
            vst1q_f16(output_data + (i<<3), vbslq_f32(c0, b, a));
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

template <bool reverse>
Status ArmWhereLayerAcc::exec_fp16_general(Blob *in_0, Blob *in_1, Blob *in_2, Blob *out) {
    fp16_t *input0_data  = reinterpret_cast<fp16_t*>(GetBlobHandlePtr(in_0->GetHandle()));
    fp16_t *input1_data  = reinterpret_cast<fp16_t*>(GetBlobHandlePtr(in_1->GetHandle()));
    int8_t *input2_data = reinterpret_cast<int8_t*>(GetBlobHandlePtr(in_2->GetHandle()));
    fp16_t *output_data  = reinterpret_cast<fp16_t*>(GetBlobHandlePtr(out->GetHandle()));

    if (!DimsVectorUtils::Equal(in_0->GetBlobDesc().dims, in_2->GetBlobDesc().dims)) {
        int8_t *broadcast = reinterpret_cast<int8_t*>(context_->GetSharedWorkSpace(DimsVectorUtils::Count(in_0->GetBlobDesc().dims) + NEON_KERNEL_EXTRA_LOAD));
        Broadcast<int8_t>(input2_data, in_2->GetBlobDesc().dims, broadcast, in_0->GetBlobDesc().dims);
        input2_data = broadcast;
    }

    if (!DimsVectorUtils::Equal(in_1->GetBlobDesc().dims, out->GetBlobDesc().dims)) {
        fp16_t *broadcast = reinterpret_cast<fp16_t*>(context_->GetSharedWorkSpace(DimsVectorUtils::Count(out->GetBlobDesc().dims) * sizeof(fp16_t) + NEON_KERNEL_EXTRA_LOAD));
        Broadcast<fp16_t>(input1_data, in_1->GetBlobDesc().dims, broadcast, out->GetBlobDesc().dims);
        input1_data = broadcast;
    }

    auto dims         = out->GetBlobDesc().dims;
    auto count        = DimsVectorUtils::Count(dims);

#ifdef TNN_USE_NEON
    int16x8_t v_zero = vdupq_n_s32(0);
#endif

    OMP_PARALLEL_FOR_
    for (int i = 0; i < UP_DIV(count, 8); ++i) {
#ifdef TNN_USE_NEON
        int16x8_t c  = vmovl_s8(vld1_s8(input2_data + (i<<3)));
        uint16x8_t c0 = vceqq_s16(c, v_zero);
        float16x8_t a = vld1q_f16(input0_data + (i<<3));
        float16x8_t b = vld1q_f16(input1_data + (i<<3));
        if (reverse) {
            vst1q_f16(output_data + (i<<3), vbslq_f16(c0, a, b));
        } else {
            vst1q_f16(output_data + (i<<3), vbslq_f32(c0, b, a));
        }
#else
        for (int j = 0; j < 8; ++j) {
            if (reverse) {
                output_data[(i<<3)+j] = input2_data[(i<<3)+j] == 0 ? input0_data[(i<<3)+j] : input1_data[(i<<3)+j];
            } else {
                output_data[(i<<3)+j] = input2_data[(i<<3)+j] == 0 ? input1_data[(i<<3)+j] : input0_data[(i<<3)+j];
            }
        }
#endif
    }

    return TNN_OK;
}

template Status ArmWhereLayerAcc::exec_fp16_single<true>(Blob *in_0, Blob *in_1, Blob *in_2, Blob *out);
template Status ArmWhereLayerAcc::exec_fp16_single<false>(Blob *in_0, Blob *in_1, Blob *in_2, Blob *out);
template Status ArmWhereLayerAcc::exec_fp16_general<true>(Blob *in_0, Blob *in_1, Blob *in_2, Blob *out);
template Status ArmWhereLayerAcc::exec_fp16_general<false>(Blob *in_0, Blob *in_1, Blob *in_2, Blob *out);

}  // namespace TNN_NS

#endif // TNN_ARM82
