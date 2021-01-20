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

#ifndef Float8_hpp
#define Float8_hpp
#include "tnn/core/macro.h"
#include "tnn/device/x86/x86_common.h"
namespace TNN_NS {

struct Float8 {
    __m256 value;
    Float8() {}
    Float8(const float v) {
        value = _mm256_set1_ps(v);
    }
    Float8(const __m256& v) {
        value = v;
    }
    Float8(const __m256&& v) {
        value = std::move(v);
    }
    Float8(const Float8& lr) {
        value = lr.value;
    }
    Float8(const Float8&& lr) {
        value = std::move(lr.value);
    }

    // void set_lane(float v, int i) {
    //     value[i] = v;
    // }

    // const float operator[](const int i) const {
    //     return value[i];
    // }

    static Float8 load(const float* addr) {
        Float8 v;
        v.value = _mm256_load_ps(addr);
        return v;
    }
    static Float8 loadu(const float* addr) {
        Float8 v;
        v.value = _mm256_loadu_ps(addr);
        return v;
    }
    static void save(float* addr, const Float8& v) {
        _mm256_store_ps(addr, v.value);
    }
    static void saveu(float* addr, const Float8& v) {
        _mm256_storeu_ps(addr, v.value);
    }
    static void mla(Float8& v1, const Float8& v2, const Float8& v3) {
        v1.value = _mm256_fmadd_ps(v2.value, v3.value, v1.value);
    }
    static void mls(Float8& v1, const Float8& v2, const Float8& v3) {
        v1.value = _mm256_fmsub_ps(v2.value, v3.value, v1.value);
    }
    static Float8 max(const Float8& v1, const Float8& v2) {
        Float8 dst;
        dst.value = _mm256_max_ps(v1.value, v2.value);
        return dst;
    }
    static Float8 min(const Float8& v1, const Float8& v2) {
        Float8 dst;
        dst.value = _mm256_min_ps(v1.value, v2.value);
        return dst;
    }
    Float8 operator+(const Float8& lr) {
        Float8 dst;
        dst.value = _mm256_add_ps(value, lr.value);
        return dst;
    }
    Float8 operator-(const Float8& lr) {
        Float8 dst;
        dst.value = _mm256_sub_ps(value, lr.value);
        return dst;
    }
    Float8 operator*(float lr) {
        Float8 dst;
        __m256 tmp = _mm256_set1_ps(lr);
        dst.value = _mm256_mul_ps(value, tmp);
        return dst;
    }
    Float8 operator*(const Float8& lr) {
        Float8 dst;
        dst.value = _mm256_mul_ps(value, lr.value);
        return dst;
    }
    Float8& operator=(const Float8& lr) {
        value = lr.value;
        return *this;
    }
    Float8& operator=(const Float8&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Float8 operator-() {
        Float8 dst;
        dst.value = _mm256_sub_ps(_mm256_setzero_ps(), value);
        return dst;
    }
};

}  // namespace TNN_NS

#endif /* Float8_hpp */
