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

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

namespace TNN_NS {

DECLARE_ARM_ACC(Equal, LAYER_EQUAL);

Status ArmEqualLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs.size() != 2) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "Equal layer's inputs size must be 2");
    }

    if (inputs[0]->GetBlobDesc().data_type != DATA_TYPE_INT32 || inputs[1]->GetBlobDesc().data_type != DATA_TYPE_INT32) {
        LOGE("Error: invalid inputs dtype\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported equal layer's inputs dtype");
    }

    if (outputs[0]->GetBlobDesc().data_type != DATA_TYPE_INT8) {
        LOGE("Error: invalid output dtype\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported equal layer's output dtype");
    }

    Blob *in_0 = inputs[0];
    Blob *in_1 = inputs[1];
    Blob *out  = outputs[0];
    if (DimsVectorUtils::Count(in_0->GetBlobDesc().dims) == 1) {
        std::swap(in_0, in_1);
    }

    if (!(DimsVectorUtils::Count(in_1->GetBlobDesc().dims) == 1)) {
        LOGE("Error: invalid input shape\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported equal layer's input shape");
    }

    if ((DimsVectorUtils::Count(in_0->GetBlobDesc().dims) != DimsVectorUtils::Count(out->GetBlobDesc().dims))) {
        LOGE("Error: mismatch input and output shape\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported equal layer's input and output shape");
    }

    int32_t *input0_data = reinterpret_cast<int32_t*>(GetBlobHandlePtr(in_0->GetHandle()));
    int32_t *input1_data = reinterpret_cast<int32_t*>(GetBlobHandlePtr(in_1->GetHandle()));
    int8_t  *output_data = reinterpret_cast<int8_t*>(GetBlobHandlePtr(out->GetHandle()));

    auto dims         = out->GetBlobDesc().dims;
    auto count        = DimsVectorUtils::Count(dims);

#ifdef TNN_USE_NEON
    int32x4_t b     = vdupq_n_s32(input1_data[0]);
    int8x8_t v_one  = vdup_n_s8(1);
    int8x8_t v_zero = vdup_n_s8(0);
#else
    int32_t b = input1_data[0];
#endif

    OMP_PARALLEL_FOR_
    for (int i = 0; i < UP_DIV(count, 8); ++i) {
#ifdef TNN_USE_NEON
        uint32x4_t c0 = vceqq_s32(vld1q_s32(input0_data + (i<<3)), b);
        uint32x4_t c1 = vceqq_s32(vld1q_s32(input0_data + (i<<3) + 4), b);
        uint16x8_t c  = vcombine_u16(vmovn_u32(c0), vmovn_u32(c1));
        int8x8_t res = vbsl_s8(vmovn_u16(c), v_one, v_zero);
        vst1_s8(output_data + (i<<3), res);
#else
        for (int j = 0; j < 8; ++j) {
            output_data[(i<<3)+j] = (int8_t)(input0_data[(i<<3)+j] == b);
        }
#endif
    }
    // vceqq_s32

    return TNN_OK;
}

REGISTER_ARM_ACC(Equal, LAYER_EQUAL);
REGISTER_ARM_LAYOUT(LAYER_EQUAL, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
