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
// y = -x1*log(x0)
// dy/dx0 = -x1/x0
// dy/dx1 = -log(x0)
class ArmCategoricalCrossEntropyGradOp : public ArmBinaryGradOp {
private:
    virtual std::pair<float, float> cal_grad(const float &i_0, const float &i_1, const float &o, const float &og) override {
        return {-i_1 / i_0 * og, -std::log(i_0) * og};
    }
    virtual std::pair<Float4, Float4> cal_grad(const Float4 &i_0, const Float4 &i_1, const Float4 &o,
                                               const Float4 &og) override {
        Float4 g0 = -Float4::div(i_1, i_0);
        Float4 g1 = -Float4::log(i_0);
        return {g0 * og, g1 * og};
    }
};

REGISTER_ARM_GRAD_OP(CategoricalCrossEntropy, LAYER_CATEGORICAL_CROSSENTROPY)
REGISTER_ARM_GRAD_LAYOUT(LAYER_CATEGORICAL_CROSSENTROPY, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
