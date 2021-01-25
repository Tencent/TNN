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

#ifndef Float4_hpp
#define Float4_hpp
#include "tnn/core/macro.h"
#include "tnn/device/x86/x86_common.h"
#include "tnn/device/x86/acc/sse_mathfun.h"

#ifdef __GNUC__
#if __GNUC__ < 5
#include "tnn/device/arm/acc/TNNVector.h"
#define VEC_NAIVE_IMPL
#endif
#endif

namespace TNN_NS {

#ifdef VEC_NAIVE_IMPL
using Float4 = TNNVector<float, 4>;
#else
struct Float4 {
    __m128 value;
    Float4() {}
    Float4(const float v) {
        value = _mm_set1_ps(v);
    }
    Float4(const float *addr) {
        value = _mm_set1_ps(*addr);
    }
    Float4(const __m128& v) {
        value = v;
    }
    Float4(const __m128&& v) {
        value = std::move(v);
    }
    Float4(const Float4& lr) {
        value = lr.value;
    }
    Float4(const Float4&& lr) {
        value = std::move(lr.value);
    }

    // void set_lane(float v, int i) {
    //     value[i] = v;
    // }

    // const float operator[](const int i) const {
    //     return value[i];
    // }

    static Float4 load(const float* addr) {
        Float4 v;
        v.value = _mm_load_ps(addr);
        return v;
    }
    static Float4 loadu(const float* addr) {
        Float4 v;
        v.value = _mm_loadu_ps(addr);
        return v;
    }
    static void save(float* addr, const Float4& v) {
        _mm_store_ps(addr, v.value);
    }
    static void saveu(float* addr, const Float4& v) {
        _mm_storeu_ps(addr, v.value);
    }
    static void mla(Float4& v1, const Float4& v2, const Float4& v3) {
        v1.value = _mm_add_ps(v1.value, _mm_mul_ps(v2.value, v3.value));
    }
    static void mls(Float4& v1, const Float4& v2, const Float4& v3) {
        v1.value = _mm_sub_ps(v1.value, _mm_mul_ps(v2.value, v3.value));
    }
    static Float4 bsl_cle(const Float4& c1, const Float4& c2, const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = _mm_blendv_ps(v1.value, v2.value, _mm_cmpge_ps(c2.value, c1.value));
        return dst;
    }
    static Float4 bsl_clt(const Float4& c1, const Float4& c2, const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = _mm_blendv_ps(v1.value, v2.value, _mm_cmpgt_ps(c2.value, c1.value));
        return dst;
    }
    static Float4 bsl_cge(const Float4& c1, const Float4& c2, const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = _mm_blendv_ps(v1.value, v2.value, _mm_cmpge_ps(c1.value, c2.value));
        return dst;
    }
    static Float4 bsl_cgt(const Float4& c1, const Float4& c2, const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = _mm_blendv_ps(v1.value, v2.value, _mm_cmpgt_ps(c1.value, c2.value));
        return dst;
    }
    static Float4 max(const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = _mm_max_ps(v1.value, v2.value);
        return dst;
    }
    static Float4 min(const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = _mm_min_ps(v1.value, v2.value);
        return dst;
    }
    static Float4 add(const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = _mm_add_ps(v1.value, v2.value);
        return dst;
    }
    static Float4 sub(const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = _mm_sub_ps(v1.value, v2.value);
        return dst;
    }
    static Float4 mul(const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = _mm_mul_ps(v1.value, v2.value);
        return dst;
    }
    static Float4 div(const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = _mm_div_ps(v1.value, v2.value);
        return dst;
    }
    static Float4 neg(const Float4 &v) {
        Float4 dst;
        _PS_CONST_TYPE(mask, uint32_t, 0x80000000);
        dst.value = _mm_xor_ps (v.value, *(__m128*) _ps_mask);
        return dst;
    }
    static Float4 abs(const Float4 &v) {
        Float4 dst;
        dst.value = _mm_max_ps(_mm_sub_ps(_mm_setzero_ps(), v.value), v.value);
        return dst;
    }
    static Float4 sqrt(const Float4 &v) {
        Float4 dst;
        dst.value = _mm_sqrt_ps(v.value);
        return dst;
    }
    static Float4 sigmoid(const Float4 &v) {
        Float4 dst;
        const __m128 one = _mm_set1_ps(1.0f);
        dst.value = _mm_div_ps(one, _mm_add_ps(one, exp_ps(_mm_sub_ps(_mm_setzero_ps(), v.value))));
        return dst;
    }
    Float4 operator+(const Float4& lr) {
        Float4 dst;
        dst.value = _mm_add_ps(value, lr.value);
        return dst;
    }
    Float4 operator-(const Float4& lr) {
        Float4 dst;
        dst.value = _mm_sub_ps(value, lr.value);
        return dst;
    }
    Float4 operator*(float lr) {
        Float4 dst;
        __m128 tmp = _mm_set1_ps(lr);
        dst.value = _mm_mul_ps(value, tmp);
        return dst;
    }
    Float4 operator*(const Float4& lr) {
        Float4 dst;
        dst.value = _mm_mul_ps(value, lr.value);
        return dst;
    }
    Float4& operator=(const Float4& lr) {
        value = lr.value;
        return *this;
    }
    Float4& operator=(const Float4&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Float4 operator-() {
        Float4 dst;
        dst.value = _mm_sub_ps(_mm_setzero_ps(), value);
        return dst;
    }
};
#endif

}  // namespace TNN_NS

#endif /* Float4_hpp */
