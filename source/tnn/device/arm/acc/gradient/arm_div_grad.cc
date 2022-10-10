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

#include "tnn/device/arm/acc/gradient/arm_binary_grad.h"

namespace TNN_NS {

// z = x / y
// dz/dx = 1 / y
// dz/dy = - x / y ^ 2
typedef struct arm_div_grad_function: arm_binary_grad_function {
    virtual std::pair<float, float> operator()(const float &i_0, const float &i_1, const float &o, const float &og) {
        return {og / i_1, -og * i_0 / (i_1 * i_1)};
    }
    virtual std::pair<Float4, Float4> operator()(const Float4 &i_0, const Float4 &i_1, const Float4 &o,
                                                 const Float4 &og) {
        return {Float4::div(og, i_1), Float4::div(-og * i_0, i_1 * i_1)};
    }
} ARM_DIV_GRAD_FUNC;

DEFINE_ARM_BINARY_GRAD_OP(Div, ARM_DIV_GRAD_FUNC)

REGISTER_ARM_GRAD_OP(Div, LAYER_DIV)
REGISTER_ARM_GRAD_LAYOUT(LAYER_DIV, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
