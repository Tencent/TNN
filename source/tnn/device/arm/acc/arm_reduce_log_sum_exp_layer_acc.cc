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

typedef struct arm_reduce_log_sum_exp_operator : arm_reduce_operator {
    virtual Float4 Calculate(Float4 &v, Float4 &t) {
        return v + Float4::exp(t);
    };

    virtual float Calculate(const float &v, const float &t) {
        return v + std::exp(t);
    };

    virtual Float4 PostCalculate(const Float4 &v, const Float4 &t) {
        return Float4::log(v);
    };

    virtual float PostCalculate(const float &v, const float &t) {
        return std::log(v);
    };
} ARM_REDUCE_LOG_SUM_EXP_OP;

DECLARE_ARM_REDUCE_ACC(ReduceLogSumExp, ARM_REDUCE_LOG_SUM_EXP_OP);

REGISTER_ARM_ACC(ReduceLogSumExp, LAYER_REDUCE_LOG_SUM_EXP);

}  // namespace TNN_NS
