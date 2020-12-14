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

#include "tnn/device/x86/acc/x86_unary_layer_acc.h"
#include <cmath>

namespace TNN_NS {
typedef struct x86_erf_operator : x86_unary_operator {
    virtual float operator()(const float x) {
        auto t = 1 / (1 + 0.5 * fabs(x));
        auto v = t * exp(-x * x - 1.26551223 + 1.00002368 * t + 0.37409196 * pow(t, 2) + 0.09678418 * pow(t, 3) -
                         0.18628806 * pow(t, 4) + 0.27886807 * pow(t, 5) - 1.13520398 * pow(t, 6) +
                         1.48851587 * pow(t, 7) - 0.82215223 * pow(t, 8) + 0.17087277 * pow(t, 9));
        if (x >= 0) {
            return 1 - v;
        } else {
            return v - 1;
        }
    }
} X86_ERF_OP;

DECLARE_X86_UNARY_ACC(Erf, X86_ERF_OP);

REGISTER_X86_ACC(Erf, LAYER_ERF);

}   // namespace TNN_NS