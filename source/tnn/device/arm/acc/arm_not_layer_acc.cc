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

DECLARE_ARM_ACC(Not, LAYER_NOT);

Status ArmNotLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs.size() != 1) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "Not layer's inputs size must be 1");
    }

    if (inputs[0]->GetBlobDesc().data_type != DATA_TYPE_INT8) {
        LOGE("Error: invalid inputs dtype\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported Not layer's inputs dtype");
    }

    if (outputs[0]->GetBlobDesc().data_type != DATA_TYPE_INT8) {
        LOGE("Error: invalid output dtype\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported Not layer's output dtype");
    }

    auto input  = inputs[0];
    auto output = outputs[0];

    int8_t *input_data  = reinterpret_cast<int8_t*>(GetBlobHandlePtr(input->GetHandle()));
    int8_t *output_data = reinterpret_cast<int8_t*>(GetBlobHandlePtr(output->GetHandle()));

    auto dims   = output->GetBlobDesc().dims;
    auto count  = DimsVectorUtils::Count(dims);

#ifdef TNN_USE_NEON
    int8x8_t b      = vdup_n_s8(0);
    int8x8_t v_one  = vdup_n_s8(1);
    int8x8_t v_zero = vdup_n_s8(0);
#else
    int8_t b = 0;
#endif

    OMP_PARALLEL_FOR_
    for (int i = 0; i < UP_DIV(count, 8); ++i) {
#ifdef TNN_USE_NEON
        uint8x8_t c = vceq_s8(vld1_s8(input_data + (i<<3)), b);
        int8x8_t res = vbsl_s8(c, v_one, v_zero);
        vst1_s8(output_data + (i<<3), res);
#else
        for (int j = 0; j < 8; ++j) {
            output_data[(i<<3)+j] = input_data[(i<<3)+j] == b ? 1 : 0;
        }
#endif
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Not, LAYER_NOT);
REGISTER_ARM_LAYOUT(LAYER_NOT, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
