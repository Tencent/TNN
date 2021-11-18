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

#include "tnn/device/arm/acc/gradient/arm_unary_grad.h"

namespace TNN_NS {

// y = sigmoid(x)
// dy/dx = y(1-y)
typedef struct arm_sigmoid_grad_operator : arm_unary_grad_operator {
    virtual float operator()(const float &i, const float &o, const float &og) {
        return o * (1.0 - o) * og;
    }
    virtual Float4 operator()(const Float4 &i, const Float4 &o, const Float4 &og) {
        return o * (Float4(1.0) - o) * og;
    }
} ARM_SIGMOID_GRAD_OP;

DEFINE_ARM_UNARY_LAYER_GRAD(Sigmoid, ARM_SIGMOID_GRAD_OP)

REGISTER_ARM_LAYER_GRAD(Sigmoid, LAYER_SIGMOID)
REGISTER_ARM_GRAD_LAYOUT(LAYER_SIGMOID, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
