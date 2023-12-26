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
class ArmSigmoidGradOp : public ArmUnaryGradOp {
private:
    virtual float cal_grad(const float &i, const float &o, const float &og) override {
        return o * (1.0 - o) * og;
    }
    virtual Float4 cal_grad(const Float4 &i, const Float4 &o, const Float4 &og) override {
        return o * (Float4(1.0) - o) * og;
    }
};

REGISTER_ARM_GRAD_OP(Sigmoid, LAYER_SIGMOID)
REGISTER_ARM_GRAD_LAYOUT(LAYER_SIGMOID, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
