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

typedef struct arm_sigmoid_operator : arm_unary_operator {
    virtual Float4 operator()(const Float4& v) {
        return Float4::sigmoid(v);
    }
    virtual Float4 fast_op(const Float4& v) {
        return Float4::fast_sigmoid(v);
    }
#ifdef TNN_ARM82
    virtual float16x8_t operator()(const float16x8_t &v) {
        Float4 v_low  = vcvt_f32_f16(vget_low_f16(v));
        Float4 v_high = vcvt_f32_f16(vget_high_f16(v));
        v_low  = Float4::sigmoid(v_low);
        v_high = Float4::sigmoid(v_high);
        float16x8_t ans = vcombine_f16(vcvt_f16_f32(v_low.value), vcvt_f16_f32(v_high.value));
        return ans;
    }
#endif
} ARM_SIGMOID_OP;

DECLARE_ARM_UNARY_ACC(Sigmoid, ARM_SIGMOID_OP);

REGISTER_ARM_ACC(Sigmoid, LAYER_SIGMOID)
REGISTER_ARM_PRECISION_FP16(LAYER_SIGMOID)

}  // namespace TNN_NS
