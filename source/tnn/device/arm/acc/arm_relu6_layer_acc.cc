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

typedef struct arm_relu6_operator : arm_unary_operator{
    virtual Float4 operator()(const Float4 &v){
        return Float4::min(Float4(6.0), Float4::max(Float4(0.0), v));
    }

#if TNN_ARM82
    virtual Half8 operator()(const Half8 &v) {
        return Half8::min(Half8(fp16_t(6.0)), Half8::max(Half8(fp16_t(0.0)), v));
    }
#endif
}
ARM_RELU6_OP;

DECLARE_ARM_UNARY_ACC(Relu6, ARM_RELU6_OP);

REGISTER_ARM_ACC(Relu6, LAYER_RELU6);
REGISTER_ARM_PRECISION_FP16(LAYER_RELU6);
}  // namespace TNN_NS
