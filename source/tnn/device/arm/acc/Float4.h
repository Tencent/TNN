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
#include <algorithm>  // supply std::max and std::min
#include <cmath>
#include "tnn/core/macro.h"
#include "tnn/utils/bfp16.h"
#include "tnn/device/arm/acc/TNNVector.h"
#ifdef TNN_USE_NEON
#include <arm_neon.h>
#include "tnn/device/arm/acc/neon_mathfun.h"
#endif
namespace TNN_NS {
#ifdef TNN_USE_NEON

struct Float2 {
    float32x2_t value;

#ifdef __aarch64__
    const float operator[](const int i) const {
        return value[i];
    }
#else
    const float operator[](const int i) const {
        float tmp_v;
        if (i == 0) {
            tmp_v = vget_lane_f32(value, 0);
        } else if (i == 1) {
            tmp_v = vget_lane_f32(value, 1);
        }
        return tmp_v;
    }
#endif
};

struct Float4 {
    float32x4_t value;
    Float4() {}
    Float4(const float v) {
        value = vdupq_n_f32(v);
    }
    Float4(const float32x4_t& v) {
        value = v;
    }
    Float4(const float32x4_t&& v) {
        value = std::move(v);
    }
    Float4(const Float4& lr) {
        value = lr.value;
    }
    Float4(const Float4&& lr) {
        value = std::move(lr.value);
    }

#ifdef __aarch64__
    void set_lane(float v, int i) {
        value[i] = v;
    }

    const float operator[](const int i) const {
        return value[i];
    }
#else
    void set_lane(float v, int i) {
        if (i == 0) {
            value = vsetq_lane_f32(v, value, 0);
        } else if (i == 1) {
            value = vsetq_lane_f32(v, value, 1);
        } else if (i == 2) {
            value = vsetq_lane_f32(v, value, 2);
        } else if (i == 3) {
            value = vsetq_lane_f32(v, value, 3);
        }
    }

    const float operator[](const int i) const {
        float tmp_v;
        if (i == 0) {
            tmp_v = vgetq_lane_f32(value, 0);
        } else if (i == 1) {
            tmp_v = vgetq_lane_f32(value, 1);
        } else if (i == 2) {
            tmp_v = vgetq_lane_f32(value, 2);
        } else if (i == 3) {
            tmp_v = vgetq_lane_f32(value, 3);
        }
        return tmp_v;
    }
#endif

    static Float4 load(const float* addr) {
        Float4 v;
        v.value = vld1q_f32(addr);
        return v;
    }
    static void save(float* addr, const Float4& v) {
        vst1q_f32(addr, v.value);
    }
    static Float4 load(const bfp16_t* addr) {
        Float4 v;
#if __aarch64__
        asm volatile(
            "ld1    {v1.4h}, [%1]\n"
            "shll   %0.4s, v1.4h, #16\n"
            : "=w"(v.value)
            : "r"(addr)
            : "memory", "v1");
#else   // __aarch64__
        asm volatile(
            "vld1.u16    d1, [%1]\n"
            "vshll.u16  %q0, d1, #16\n"
            : "=w"(v.value)
            : "r"(addr)
            : "memory", "v1");
#endif  // __aarch64__
        return v;
    }
    static void save(bfp16_t* addr, const Float4& v) {
#if __aarch64__
        asm volatile(
            "shrn   v1.4h, %1.4s, #16\n"
            "st1    {v1.4h}, [%0]\n"
            : "+r"(addr)
            : "w"(v.value)
            : "memory", "v1");
#else   // __aarch64__
        asm volatile(
            "vshrn.u32  d1, %q1, #16\n"
            "vst1.u16    d1, [%0]\n"
            : "+r"(addr)
            : "w"(v.value)
            : "memory", "v1");
#endif  // __aarch64__
    }
    static void get_low(Float4& v1, Float2& v2) {
        v2.value = vget_low_f32(v1.value);
    }
    static void get_high(Float4& v1, Float2& v2) {
        v2.value = vget_high_f32(v1.value);
    }
    static Float4 combine(Float2& v1, Float2& v2) {
        return vcombine_f32(v1.value, v2.value);
    }
    static Float4 extract(const Float4& v1, const Float4& v2, const int n) {
        Float4 dst;
        if (n == 0) {
            dst.value = v1.value;
        } else if (n == 1) {
            dst.value = vextq_f32(v1.value, v2.value, 1);
        } else if (n == 2) {
            dst.value = vextq_f32(v1.value, v2.value, 2);
        } else if (n == 3) {
            dst.value = vextq_f32(v1.value, v2.value, 3);
        } else if (n == 4) {
            dst.value = v2.value;
        }
        return dst;
    }
    static Float4 pad(const Float4& v1, const Float4& v2, const int n) {
        static const uint32_t select  = uint32_t(-1);
        static const uint32x4_t mask1 = {select,select,select,0};
        static const uint32x4_t mask2 = {select,select,0,0};
        static const uint32x4_t mask3 = {select,0,0,0};
        Float4 dst;
        if (n == 0) {
            dst.value = v1.value;
        } else if (n == 1) {
            dst.value = vbslq_f32(mask1, v1.value, v2.value);
        } else if (n == 2) {
            dst.value = vbslq_f32(mask2, v1.value, v2.value);
        } else if (n == 3) {
            dst.value =  vbslq_f32(mask3, v1.value, v2.value);
        } else if (n == 4) {
            dst.value = v2.value;
        }
        return dst;
    }
    static void mla(Float4& v1, const Float4& v2, const Float4& v3) {
        v1.value = vmlaq_f32(v1.value, v2.value, v3.value);
    }
    static void mla_lane0(Float4& v1, const Float4& v2, const Float2& v3) {
        v1.value = vmlaq_lane_f32(v1.value, v2.value, v3.value, 0);
    }
    static void mla_lane1(Float4& v1, const Float4& v2, const Float2& v3) {
        v1.value = vmlaq_lane_f32(v1.value, v2.value, v3.value, 1);
    }
    static void mls(Float4& v1, const Float4& v2, const Float4& v3) {
        v1.value = vmlsq_f32(v1.value, v2.value, v3.value);
    }
    static void mls_lane0(Float4& v1, const Float4& v2, const Float2& v3) {
        v1.value = vmlsq_lane_f32(v1.value, v2.value, v3.value, 0);
    }
    static void mls_lane1(Float4& v1, const Float4& v2, const Float2& v3) {
        v1.value = vmlsq_lane_f32(v1.value, v2.value, v3.value, 1);
    }
    static Float4 bsl_cle(const Float4& c1, const Float4& c2, const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = vbslq_f32(vcleq_f32(c1.value, c2.value), v1.value, v2.value);
        return dst;
    }
    static Float4 bsl_clt(const Float4& c1, const Float4& c2, const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = vbslq_f32(vcltq_f32(c1.value, c2.value), v1.value, v2.value);
        return dst;
    }
    static Float4 bsl_cge(const Float4& c1, const Float4& c2, const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = vbslq_f32(vcgeq_f32(c1.value, c2.value), v1.value, v2.value);
        return dst;
    }
    static Float4 bsl_cgt(const Float4& c1, const Float4& c2, const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = vbslq_f32(vcgtq_f32(c1.value, c2.value), v1.value, v2.value);
        return dst;
    }
    static Float4 neg(const Float4& v) {
        Float4 dst;
        dst.value = vnegq_f32(v.value);
        return dst;
    }
    static Float4 floor(const Float4& v) {
        Float4 dst;
#if __aarch64__
        dst.value = vcvtq_f32_s32(vcvtmq_s32_f32(v.value));
#else
        int32x4_t s32   = vcvtq_s32_f32(v.value);
        uint32x4_t mask = vcgtq_f32(vcvtq_f32_s32(s32), v.value);
        dst.value       = vcvtq_f32_s32(vaddq_s32(s32, vreinterpretq_s32_u32(mask)));
#endif
        return dst;
    }
    static Float4 ceil(const Float4& v) {
        Float4 dst;
#if __aarch64__
        dst.value = vcvtq_f32_s32(vcvtpq_s32_f32(v.value));
#else
        int32x4_t s32   = vcvtq_s32_f32(v.value);
        uint32x4_t mask = vcgtq_f32(v.value, vcvtq_f32_s32(s32));
        dst.value       = vcvtq_f32_s32(vsubq_s32(s32, vreinterpretq_s32_u32(mask)));
#endif
        return dst;
    }
    static Float4 max(const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = vmaxq_f32(v1.value, v2.value);
        return dst;
    }
    static Float4 min(const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = vminq_f32(v1.value, v2.value);
        return dst;
    }
    static Float4 div(const Float4& v1, const Float4& v2) {
        Float4 dst;
        dst.value = div_ps(v1.value, v2.value);
        return dst;
    }
    static Float4 exp(const Float4& v) {
        Float4 dst;
        dst.value = exp_ps(v.value);
        return dst;
    }
    static Float4 pow(const Float4& v, const Float4& e) {
        Float4 dst;
        dst.value = pow_ps(v.value, e.value);
        return dst;
    }
    static Float4 sqrt(const Float4& v) {
        Float4 dst;
        static float32x4_t zero = vdupq_n_f32(0.0f);
        dst.value = vbslq_f32(vceqq_f32(v.value, zero), zero, sqrt_ps(v.value));
        return dst;
    }
    static Float4 tanh(const Float4& v) {
        Float4 dst;
        dst.value = tanh_ps(v.value);
        return dst;
    }
    static Float4 tan(const Float4& v) {
        Float4 dst;

        float32x4_t ysin, ycos;
        sincos_ps(v.value, &ysin, &ycos);
        dst.value = div_ps(ysin, ycos);
        return dst;
    }
    static Float4 sin(const Float4& v) {
        Float4 dst;
        dst.value = sin_ps(v.value);
        return dst;
    }
    static Float4 cos(const Float4& v) {
        Float4 dst;
        dst.value = cos_ps(v.value);
        return dst;
    }
    static Float4 sigmoid(const Float4& v) {
        Float4 dst;
        dst.value = sigmoid_ps(v.value);
        return dst;
    }
    static Float4 fast_sigmoid(const Float4& v) {
        Float4 dst;
        dst.value = fast_sigmoid_ps(v.value);
        return dst;
    }
    static Float4 log(const Float4& v) {
        Float4 dst;
        dst.value = log_ps(v.value);
        return dst;
    }
    static Float4 abs(const Float4& v) {
        Float4 dst;
        dst.value = vabsq_f32(v.value);
        return dst;
    }
    Float4 operator+(const Float4& lr) const {
        Float4 dst;
        dst.value = value + lr.value;
        return dst;
    }
    Float4 operator-(const Float4& lr) const {
        Float4 dst;
        dst.value = value - lr.value;
        return dst;
    }
    Float4 operator*(float lr) const {
        Float4 dst;
        dst.value = vmulq_n_f32(value, lr);
        return dst;
    }
    Float4 operator*(const Float4& lr) const {
        Float4 dst;
        dst.value = value * lr.value;
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
    Float4 operator-() const {
        Float4 dst;
        dst.value = -value;
        return dst;
    }
};

struct Float4x4 {
    float32x4x4_t value;
    static Float4x4 ld4(const float* addr) {
        Float4x4 v;
        v.value = vld4q_f32(addr);
        return v;
    }
    static Float4x4 load(const float* addr) {
        Float4x4 v;
        v.value.val[0] = vld1q_f32(addr);
        v.value.val[1] = vld1q_f32(addr + 4);
        v.value.val[2] = vld1q_f32(addr + 8);
        v.value.val[3] = vld1q_f32(addr + 12);
        return v;
    }
    static void save(float* addr, const Float4x4& v) {
        vst1q_f32(addr, v.value.val[0]);
        vst1q_f32(addr + 4, v.value.val[1]);
        vst1q_f32(addr + 8, v.value.val[2]);
        vst1q_f32(addr + 12, v.value.val[3]);
    }

    static Float4x4 max(const Float4x4& v1, const Float4& v2) {
        Float4x4 dst;
        dst.value.val[0] = vmaxq_f32(v1.value.val[0], v2.value);
        dst.value.val[1] = vmaxq_f32(v1.value.val[1], v2.value);
        dst.value.val[2] = vmaxq_f32(v1.value.val[2], v2.value);
        dst.value.val[3] = vmaxq_f32(v1.value.val[3], v2.value);
        return dst;
    }
    static Float4x4 min(const Float4x4& v1, const Float4& v2) {
        Float4x4 dst;
        dst.value.val[0] = vminq_f32(v1.value.val[0], v2.value);
        dst.value.val[1] = vminq_f32(v1.value.val[1], v2.value);
        dst.value.val[2] = vminq_f32(v1.value.val[2], v2.value);
        dst.value.val[3] = vminq_f32(v1.value.val[3], v2.value);
        return dst;
    }

    Float4x4 operator+(const Float4& v2) {
        Float4x4 dst;
        dst.value.val[0] = value.val[0] + v2.value;
        dst.value.val[1] = value.val[1] + v2.value;
        dst.value.val[2] = value.val[2] + v2.value;
        dst.value.val[3] = value.val[3] + v2.value;
        return dst;
    }

    static Float4x4 load(const bfp16_t* addr) {
        Float4x4 v;
        asm volatile(
#if __aarch64__
            "ld1    {v1.4s, v2.4s}, [%4]\n"
            "shll   %0.4s, v1.4h, #16\n"
            "shll   %2.4s, v2.4h, #16\n"
            "shll2   %1.4s, v1.8h, #16\n"
            "shll2   %3.4s, v2.8h, #16\n"
            : "=w"(v.value.val[0]), "=w"(v.value.val[1]), "=w"(v.value.val[2]), "=w"(v.value.val[3])
            : "r"(addr)
            : "memory", "v1", "v2");
#else
            "vld1.32 {q1, q2}, [%4]\n"
            "vshll.u16  %q0, d2, #16\n"
            "vshll.u16  %q1, d3, #16\n"
            "vshll.u16  %q2, d4, #16\n"
            "vshll.u16  %q3, d5, #16\n"
            : "=w"(v.value.val[0]), "=w"(v.value.val[1]), "=w"(v.value.val[2]), "=w"(v.value.val[3])
            : "r"(addr)
            : "memory", "q1", "q2");
#endif  // __aarch64__
        return v;
    }

    static void save(bfp16_t* addr, const Float4x4& v) {
        asm volatile(
#if __aarch64__
            "shrn   v4.4h, %1.4s, #16\n"
            "shrn   v5.4h, %2.4s, #16\n"
            "shrn   v6.4h, %3.4s, #16\n"
            "shrn   v7.4h, %4.4s, #16\n"
            "st1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%0]\n"
            :
            : "r"(addr), "w"(v.value.val[0]), "w"(v.value.val[1]), "w"(v.value.val[2]), "w"(v.value.val[3])
            : "memory", "v4", "v5", "v6", "v7");
#else   //
            "vshrn.u32  d8, %q1, #16\n"
            "vshrn.u32  d9, %q2, #16\n"
            "vshrn.u32  d10, %q3, #16\n"
            "vshrn.u32  d11, %q4, #16\n"
            "vst1.u16    {q4, q5}, [%0]\n"
            :
            : "r"(addr), "w"(v.value.val[0]), "w"(v.value.val[1]), "w"(v.value.val[2]), "w"(v.value.val[3])
            : "memory", "q4", "q5");
#endif  // __aarch64__
    }
    static void st1_lane(float* addr, const Float4x4& v, int index) {
        vst1q_f32(addr, v.value.val[index]);
    }

    void get_lane(Float4& v, int index) {
        v.value = value.val[index];
    }
};

struct Short4x4 {
    int16x4x4_t value;
    static Short4x4 ld4(const int16_t* addr) {
        Short4x4 v;
        v.value = vld4_s16(addr);
        return v;
    }
    static void st1_lane(int16_t* addr, const Short4x4& v, int index) {
        vst1_s16(addr, v.value.val[index]);
    }
};

#else

struct Float2 : TNNVector<float, 2> {
    using TNNVector<float, 2>::TNNVector;
};

struct Float4 : TNNVector<float, 4> {
    using TNNVector<float, 4>::TNNVector;
    Float4() {}
    Float4(const Float4& lr) {
        for (int i = 0; i < 4; ++i) {
            value[i] = lr.value[i];
        }
    }
    Float4(const TNNVector<float, 4>& lr) {
        for (int i = 0; i < 4; ++i) {
            value[i] = lr.value[i];
        }
    }

    static void get_low(const Float4& v1, Float2& v2) {
        v2.value[0] = v1.value[0];
        v2.value[1] = v1.value[1];
    }
    static void get_high(const Float4& v1, Float2& v2) {
        v2.value[0] = v1.value[2];
        v2.value[1] = v1.value[3];
    }
    static Float4 combine(const Float2& v1, const Float2& v2) {
        Float4 dst;
        dst.value[0] = v1.value[0];
        dst.value[1] = v1.value[1];
        dst.value[2] = v2.value[0];
        dst.value[3] = v2.value[1];
        return dst;
    }
    static void mla_lane0(Float4& v1, const Float4& v2, const Float2& v3) {
        for (int i = 0; i < 4; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[0];
        }
    }
    static void mla_lane1(Float4& v1, const Float4& v2, const Float2& v3) {
        for (int i = 0; i < 4; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[1];
        }
    }
    static void mls_lane0(Float4& v1, const Float4& v2, const Float2& v3) {
        for (int i = 0; i < 4; ++i) {
            v1.value[i] = v1.value[i] - v2.value[i] * v3.value[0];
        }
    }
    static void mls_lane1(Float4& v1, const Float4& v2, const Float2& v3) {
        for (int i = 0; i < 4; ++i) {
            v1.value[i] = v1.value[i] - v2.value[i] * v3.value[1];
        }
    }

    using TNNVector<float, 4>::load;
    using TNNVector<float, 4>::save;
    static Float4 load(const bfp16_t* addr) {
        Float4 v;
        for (int i = 0; i < 4; ++i) {
            v.value[i] = static_cast<float>(addr[i]);
        }
        return v;
    }
    static void save(bfp16_t* addr, const Float4& v) {
        for (int i = 0; i < 4; ++i) {
            addr[i] = static_cast<bfp16_t>(v.value[i]);
        }
    }
};

struct Short4 {
    int16_t value[4];
    static void save(int16_t* addr, const Short4& v) {
        for (int i = 0; i < 4; ++i) {
            addr[i] = v.value[i];
        }
    }
};

struct Short4x4 {
    Short4 value[4];
    static Short4x4 ld4(const int16_t* addr) {
        Short4x4 v;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                v.value[j].value[i] = static_cast<float>(addr[i * 4 + j]);
            }
        }
        return v;
    }

    static void st1_lane(int16_t* addr, const Short4x4& v, int index) {
        Short4::save(addr, v.value[index]);
    }
};

struct Float4x4 {
    Float4 value[4];
    static Float4x4 ld4(const float* addr) {
        Float4x4 v;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                v.value[j].value[i] = static_cast<float>(addr[i * 4 + j]);
            }
        }
        return v;
    }

    static Float4x4 max(const Float4x4& v1, const Float4& v2) {
        Float4x4 dst;
        dst.value[0] = Float4::max(v1.value[0], v2);
        dst.value[1] = Float4::max(v1.value[1], v2);
        dst.value[2] = Float4::max(v1.value[2], v2);
        dst.value[3] = Float4::max(v1.value[3], v2);
        return dst;
    }
    static Float4x4 min(const Float4x4& v1, const Float4& v2) {
        Float4x4 dst;
        dst.value[0] = Float4::min(v1.value[0], v2);
        dst.value[1] = Float4::min(v1.value[1], v2);
        dst.value[2] = Float4::min(v1.value[2], v2);
        dst.value[3] = Float4::min(v1.value[3], v2);
        return dst;
    }

    Float4x4 operator+(const Float4& v2) {
        Float4x4 dst;
        dst.value[0] = value[0] + v2;
        dst.value[1] = value[1] + v2;
        dst.value[2] = value[2] + v2;
        dst.value[3] = value[3] + v2;
        return dst;
    }
    template <typename T>
    static Float4x4 load(const T* addr) {
        Float4x4 v;
        v.value[0] = Float4::load(addr);
        v.value[1] = Float4::load(addr + 4);
        v.value[2] = Float4::load(addr + 8);
        v.value[3] = Float4::load(addr + 12);
        return v;
    }

    template <typename T>
    static void save(T* addr, const Float4x4& v) {
        Float4::save(addr, v.value[0]);
        Float4::save(addr + 4, v.value[1]);
        Float4::save(addr + 8, v.value[2]);
        Float4::save(addr + 12, v.value[3]);
    }
    static void st1_lane(float* addr, const Float4x4& v, int index) {
        Float4::save(addr, v.value[index]);
    }

    void get_lane(Float4 &v, int index) {
        v = value[index];
    }
};

#endif
}  // namespace TNN_NS

#endif /* Float4_hpp */
