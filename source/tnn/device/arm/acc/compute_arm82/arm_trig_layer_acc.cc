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

/// unary tanh op
typedef struct arm_tanh_operator : arm_unary_operator {
    virtual Float4 operator()(const Float4& v) {
        return Float4::tanh(v);
    }
} ARM_TANH_OP;

/// unary tan op
typedef struct arm_tan_operator : arm_unary_operator {
    virtual Float4 operator()(const Float4& v) {
        return Float4::tan(v);
    }
} ARM_TAN_OP;

/// unary atan op
typedef struct arm_atan_operator : arm_unary_operator {
    virtual Float4 operator()(const Float4& v) {
        Float4 ret;
        ret.set_lane(atan(v[0]), 0);
        ret.set_lane(atan(v[1]), 1);
        ret.set_lane(atan(v[2]), 2);
        ret.set_lane(atan(v[3]), 3);
        return ret;
    }
} ARM_ATAN_OP;

/// unary sin op
typedef struct arm_sin_operator : arm_unary_operator {
    virtual Float4 operator()(const Float4& v) {
        return Float4::sin(v);
    }
} ARM_SIN_OP;

/// unary asin op
typedef struct arm_asin_operator : arm_unary_operator {
    virtual Float4 operator()(const Float4& v) {
        Float4 ret;
        ret.set_lane(asin(v[0]), 0);
        ret.set_lane(asin(v[1]), 1);
        ret.set_lane(asin(v[2]), 2);
        ret.set_lane(asin(v[3]), 3);
        return ret;
    }
} ARM_ASIN_OP;

/// unary cos op
typedef struct arm_cos_operator : arm_unary_operator {
    virtual Float4 operator()(const Float4& v) {
        return Float4::cos(v);
    }
} ARM_COS_OP;

/// unary acos op
typedef struct arm_acos_operator : arm_unary_operator {
    virtual Float4 operator()(const Float4& v) {
        Float4 ret;
        ret.set_lane(acos(v[0]), 0);
        ret.set_lane(acos(v[1]), 1);
        ret.set_lane(acos(v[2]), 2);
        ret.set_lane(acos(v[3]), 3);
        return ret;
    }
} ARM_ACOS_OP;

/// register ops
DECLARE_ARM_UNARY_ACC(Tanh, ARM_TANH_OP);
REGISTER_ARM_ACC(Tanh, LAYER_TANH);

DECLARE_ARM_UNARY_ACC(Tan, ARM_TAN_OP);
REGISTER_ARM_ACC(Tan, LAYER_TAN);

DECLARE_ARM_UNARY_ACC(Atan, ARM_ATAN_OP);
REGISTER_ARM_ACC(Atan, LAYER_ATAN);

DECLARE_ARM_UNARY_ACC(Sin, ARM_SIN_OP);
REGISTER_ARM_ACC(Sin, LAYER_SIN);

DECLARE_ARM_UNARY_ACC(Asin, ARM_ASIN_OP);
REGISTER_ARM_ACC(Asin, LAYER_ASIN);

DECLARE_ARM_UNARY_ACC(Cos, ARM_COS_OP);
REGISTER_ARM_ACC(Cos, LAYER_COS);

DECLARE_ARM_UNARY_ACC(Acos, ARM_ACOS_OP);
REGISTER_ARM_ACC(Acos, LAYER_ACOS);

}  // namespace TNN_NS
