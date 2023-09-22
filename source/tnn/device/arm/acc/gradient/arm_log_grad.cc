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

// y = log(x)
// dy/dx = 1/x
typedef struct arm_log_grad_function : arm_unary_grad_function {
    virtual float operator()(const float &i, const float &o, const float &og) {
        return og / i;
    }
    virtual Float4 operator()(const Float4 &i, const Float4 &o, const Float4 &og) {
        return Float4::div(og, i);
    }
} ARM_LOG_GRAD_FUNC;

DEFINE_ARM_UNARY_GRAD_OP(Log, ARM_LOG_GRAD_FUNC)

REGISTER_ARM_GRAD_OP(Log, LAYER_LOG)
REGISTER_ARM_GRAD_LAYOUT(LAYER_LOG, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
