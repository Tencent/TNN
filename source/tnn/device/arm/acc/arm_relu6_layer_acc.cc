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

#include "tnn/device/arm/acc/arm_unary_layer_acc.h"

namespace TNN_NS {

// DECLARE_ARM_ACC(Relu6, LAYER_RELU6);

// Status ArmRelu6LayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
//     auto input  = inputs[0];
//     auto output = outputs[0];

//     auto dims = output->GetBlobDesc().dims;

//     int count      = dims[0] * ROUND_UP(dims[1], 4) * dims[2] * dims[3];
//     int count_quad = UP_DIV(count, 4);

//     if (input->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
//         auto input_ptr  = reinterpret_cast<float *>(GetBlobHandlePtr(input->GetHandle()));
//         auto output_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(output->GetHandle()));

//         for (int n = 0; n < count_quad; n++) {
//             Float4::save(output_ptr + n * 4,
//                          Float4::min(Float4(6.0), Float4::max(Float4(0.0), Float4::load(input_ptr + n * 4))));
//         }
//     } else {
//         LOGE("Error: layer acc dont support datatype: %d\n", output->GetBlobDesc().data_type);
//     }

//     return TNN_OK;
// }

// REGISTER_ARM_ACC(Relu6, LAYER_RELU6)

typedef struct arm_relu6_operator : arm_unary_operator {
    virtual Float4 operator()(const Float4 &v) {
        return Float4::min(Float4(6.0), Float4::max(Float4(0.0), v));
    }
} ARM_RELU6_OP;

DECLARE_ARM_UNARY_ACC(Relu6, ARM_RELU6_OP);

REGISTER_ARM_ACC(Relu6, LAYER_RELU6);
}  // namespace TNN_NS
