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

// y = relu(x)
// dy/dx = x > 0 ? 1 : 0
typedef struct arm_relu_grad_operator : arm_unary_grad_operator {
    virtual float operator()(const float &i, const float &o, const float &og) {
        return (i > 0.0 ? og : 0.0);
    }
    virtual Float4 operator()(const Float4 &i, const Float4 &o, const Float4 &og) {
        return Float4::bsl_cgt(i, Float4(0.0), og, Float4(0.0));
    }
} ARM_RELU_GRAD_OP;

DEFINE_ARM_UNARY_LAYER_GRAD(Relu, ARM_RELU_GRAD_OP)

REGISTER_ARM_LAYER_GRAD(Relu, LAYER_RELU)
REGISTER_ARM_GRAD_LAYOUT(LAYER_RELU, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
