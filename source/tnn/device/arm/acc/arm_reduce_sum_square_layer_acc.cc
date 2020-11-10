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

#include "tnn/device/arm/acc/arm_reduce_layer_acc.h"

namespace TNN_NS {

typedef struct arm_reduce_sum_square_operator : arm_reduce_operator {
    virtual Float4 PreCalculate(Float4 &v) {
        return v * v;
    };

    virtual float PreCalculate(const float &v) {
        return v * v;
    };

    virtual bool NeedPreCalculate() {
        return true;
    };
} ARM_REDUCE_SUM_SQUARE_OP;

DECLARE_ARM_REDUCE_ACC(ReduceSumSquare, ARM_REDUCE_SUM_SQUARE_OP);

REGISTER_ARM_ACC(ReduceSumSquare, LAYER_REDUCE_SUM_SQUARE);

}  // namespace TNN_NS
