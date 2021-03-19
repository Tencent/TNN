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

#include "tnn/device/x86/acc/x86_unary2_layer_acc.h"

#include <cmath>
#include <algorithm>

namespace TNN_NS {
typedef struct x86_sigmoid_operator : x86_unary2_operator {
    virtual float operator()(const float v) {
        return 1.0f / (1.0f + exp(-v));
    }

    virtual Float4 operator()(const Float4 &v) {
        return Float4::sigmoid(v);
    }

    virtual Float8 operator()(const Float8 &v) {
        return Float8::sigmoid(v);
    }
} X86_SIGMOID_OP;

X86_REGISTER_UNARY2_KERNEL(LAYER_SIGMOID, avx2, unary2_kernel_avx<X86_SIGMOID_OP>);
X86_REGISTER_UNARY2_KERNEL(LAYER_SIGMOID, sse42, unary2_kernel_sse<X86_SIGMOID_OP>);
DECLARE_X86_UNARY2_ACC(Sigmoid, LAYER_SIGMOID);
REGISTER_X86_ACC(Sigmoid, LAYER_SIGMOID);

}  // namespace TNN_NS
