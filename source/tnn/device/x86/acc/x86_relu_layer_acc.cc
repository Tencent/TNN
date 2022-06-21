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

#include "tnn/device/x86/acc/x86_relu_layer_acc.h"
#include "tnn/device/x86/acc/compute/x86_compute_int8.h"

#include <cmath>
#include <algorithm>

namespace TNN_NS {
typedef struct x86_relu_operator : x86_unary2_operator {
    virtual float operator()(const float v) {
        return std::max(v, 0.0f);
    }

    virtual Float4 operator()(const Float4 &v) {
        return Float4::max(v, Float4(0.f));
    }

    virtual Float8 operator()(const Float8 &v) {
        return (Float8::max(v, Float8(0.f)));
    }
} X86_RELU_OP;

X86_REGISTER_UNARY2_KERNEL(LAYER_RELU, avx2, unary2_kernel_avx<X86_RELU_OP>);
X86_REGISTER_UNARY2_KERNEL(LAYER_RELU, sse42, unary2_kernel_sse<X86_RELU_OP>);

Status X86ReluLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        auto dims = inputs[0]->GetBlobDesc().dims;
        long count = dims[0] * ROUND_UP(dims[1], 4) * DimsVectorUtils::Count(dims, 2);
        X86ReluInt8(handle_ptr<int8_t *>(outputs[0]->GetHandle()),
                    handle_ptr<int8_t *>(inputs[0]->GetHandle()), count);
        return TNN_OK;
    } else {
        return X86Unary2LayerAcc::DoForward(inputs, outputs);
    }
}

REGISTER_X86_ACC(Relu, LAYER_RELU);
}   // namespace TNN_NS