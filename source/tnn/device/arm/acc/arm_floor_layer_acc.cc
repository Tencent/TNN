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

typedef struct arm_floor_operator : arm_unary_operator {
    virtual Float4 operator()(const Float4& v) {
        return Float4::floor(v);
    }
} ARM_FLOOR_OP;

DECLARE_ARM_UNARY_ACC(Floor, ARM_FLOOR_OP);
REGISTER_ARM_ACC(Floor, LAYER_FLOOR);

typedef struct arm_ceil_operator : arm_unary_operator {
    virtual Float4 operator()(const Float4& v) {
        return Float4::ceil(v);
    }
} ARM_CEIL_OP;

DECLARE_ARM_UNARY_ACC(Ceil, ARM_CEIL_OP);
REGISTER_ARM_ACC(Ceil, LAYER_CEIL);

}  // namespace TNN_NS
