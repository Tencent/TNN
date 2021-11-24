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

// x0 is logits, x1 is true labels
// y = -x1*log(x0) - (1-x1)*log(1-x0)
// dy/dx0 = (1-x1)/(1-x0) -x1/x0
// dy/dx1 = log(1-x0) - log(x0)
typedef struct arm_bce_grad_function: arm_binary_grad_function {
    virtual std::pair<float, float> operator()(const float &i_0, const float &i_1, const float &o, const float &og) {
        return {((1.0 - i_1) / (1.0 - i_0) - i_1 / i_0) * og, (std::log(1.0 - i_0) - std::log(i_0)) * og};
    }
    virtual std::pair<Float4, Float4> operator()(const Float4 &i_0, const Float4 &i_1, const Float4 &o,
                                                 const Float4 &og) {
        Float4 g0 = Float4::div(Float4(1.0) - i_1, Float4(1.0) - i_0) - Float4::div(i_1, i_0);
        Float4 g1 = Float4::log(Float4(1.0) - i_0) - Float4::log(i_0);
        return {g0 * og, g1 * og};
    }
} ARM_BCE_GRAD_FUNC;

DEFINE_ARM_BINARY_GRAD_OP(BinaryCrossEntropy, ARM_BCE_GRAD_FUNC)

REGISTER_ARM_GRAD_OP(BinaryCrossEntropy, LAYER_BINARY_CROSSENTROPY)
REGISTER_ARM_GRAD_LAYOUT(LAYER_BINARY_CROSSENTROPY, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
