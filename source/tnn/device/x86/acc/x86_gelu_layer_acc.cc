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

#include "tnn/device/x86/acc/x86_unary2_layer_acc.h"

#include <cmath>
#include <algorithm>

namespace TNN_NS {

template <typename VEC>
static VEC fast_erf_approximation(const VEC x) {
    auto t = VEC::div(VEC(1.f), VEC(1.f) + VEC(0.5f) * VEC::abs(x));
    auto t_2 = t * t;
    auto t_3 = t_2 * t;
    auto t_4 = t_3 * t;
    auto t_5 = t_4 * t;
    auto t_6 = t_5 * t;
    auto t_7 = t_6 * t;
    auto t_8 = t_7 * t;
    auto t_9 = t_8 * t;

    auto v = t * VEC::exp(VEC::neg(x) * x - VEC(1.26551223) +
                             VEC(1.00002368) * t +
                             VEC(0.37409196) * t_2 +
                             VEC(0.09678418) * t_3 -
                             VEC(0.18628806) * t_4 +
                             VEC(0.27886807) * t_5 -
                             VEC(1.13520398) * t_6 +
                             VEC(1.48851587) * t_7 -
                             VEC(0.82215223) * t_8 +
                             VEC(0.17087277) * t_9);
    auto v_pos = VEC(1.f) - v;
    auto v_neg = v - VEC(1.f);

    return VEC::bsl_cge(x, VEC(0.f), v_pos, v_neg);
}

typedef struct x86_gelu_operator : x86_unary2_operator {
    virtual float operator()(const float v) {
        return 0.5f * v * (erff(v * 0.707106793288165f) + 1.0f);
    }

    virtual Float4 operator()(const Float4 &v) {
        return Float4(0.5f) * v * (fast_erf_approximation<Float4>(v * Float4(0.707106793288165f)) + Float4(1.f));
    }

    virtual Float8 operator()(const Float8 &v) {
        return Float8(0.5f) * v * (fast_erf_approximation<Float8>(v * Float8(0.707106793288165f)) + Float8(1.f));
    }
} X86_GELU_OP;

X86_REGISTER_UNARY2_KERNEL(LAYER_GELU, avx2, unary2_kernel_avx<X86_GELU_OP>);
X86_REGISTER_UNARY2_KERNEL(LAYER_GELU, sse42, unary2_kernel_sse<X86_GELU_OP>);
DECLARE_X86_UNARY2_ACC(Gelu, LAYER_GELU);
REGISTER_X86_ACC(Gelu, LAYER_GELU);

}   // namespace TNN_NS