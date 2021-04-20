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

#include "tnn/device/arm/acc/arm_relu_layer_acc.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

Status ArmReluLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims = output->GetBlobDesc().dims;

    long count = dims[0] * ROUND_UP(dims[1], 4) * DimsVectorUtils::Count(dims, 2);

    auto &data_type = input->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_INT8) {
        ReluInt8(reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle())),
                 reinterpret_cast<int8_t *>(GetBlobHandlePtr(input->GetHandle())), count);
    } else if (data_type == DATA_TYPE_FLOAT) {
        auto dst = reinterpret_cast<float *>(GetBlobHandlePtr(output->GetHandle()));
        auto src = reinterpret_cast<float *>(GetBlobHandlePtr(input->GetHandle()));
        Float4 vzero(0.f);
        for (long i = 0; i < count; i += 4) {
            Float4::save(dst + i, Float4::max(Float4::load(src + i), vzero));
        }
    } else if (data_type == DATA_TYPE_BFP16) {
        auto dst = reinterpret_cast<bfp16_t *>(GetBlobHandlePtr(output->GetHandle()));
        auto src = reinterpret_cast<bfp16_t *>(GetBlobHandlePtr(input->GetHandle()));
        Float4 vzero(0.f);
        for (long i = 0; i < count; i += 4) {
            Float4::save(dst + i, Float4::max(Float4::load(src + i), vzero));
        }
    }
#if TNN_ARM82
    else if (data_type == DATA_TYPE_HALF) {
        ExecFp16(inputs, outputs);
    }
#endif
    else {
        return TNNERR_LAYER_ERR;
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Relu, LAYER_RELU)
REGISTER_ARM_PRECISION_FP16(LAYER_RELU)
REGISTER_ARM_LAYOUT(LAYER_RELU, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
