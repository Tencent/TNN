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

#ifndef Half8_hpp
#define Half8_hpp
#include <algorithm>  // supply std::max and std::min
#include <cmath>
#include "tnn/core/macro.h"
#include "tnn/utils/half.hpp"
#include "tnn/utils/half_utils_inner.h"
#include "tnn/device/arm/acc/TNNVector.h"
#include "tnn/device/arm/acc/Float4.h"
#ifdef TNN_USE_NEON
#include <arm_neon.h>
#include "tnn/device/arm/acc/neon_mathfun.h"
#endif

namespace TNN_NS {

#ifdef TNN_ARM82_A64

struct Half4 {
    float16x4_t value;
    const __fp16 operator[](const int i) const {
        return value[i];
    }
    Half4() {}
    Half4(const float16x4_t& v) {
        value = v;
    }
    Half4(const float16x4_t&& v) {
        value = std::move(v);
    }
    Half4(const Half4& lr) {
        value = lr.value;
    }
    Half4(const Half4&& lr) {
        value = std::move(lr.value);
    }
    Half4(const Float4& lr) {
        value = vcvt_f16_f32(lr.value);
    }
    static Half4 load(const __fp16* addr) {
        Half4 v;
        v.value = vld1_f16(addr);
        return v;
    }
    static void save(__fp16* addr, const Half4& v) {
        vst1_f16(addr, v.value);
    }
    static void zip(Half4& v1, Half4& v2) {
        float16x4x2_t v = vzip_f16(v1.value, v2.value);
        v1.value = v.val[0];
        v2.value = v.val[1];
    }
    static void add_to_f32(Half4& v1, Float4& v2) {
        v2.value = vaddq_f32(v2.value, vcvt_f32_f16(v1.value));
    }
    Half4& operator=(const Half4& lr) {
        value = lr.value;
        return *this;
    }
};

struct Half8 {
    float16x8_t value;
    Half8() {}
    Half8(const __fp16 v) {
        value = vdupq_n_f16(v);
    }
    Half8(const float16x8_t& v) {
        value = v;
    }
    Half8(const float16x8_t&& v) {
        value = std::move(v);
    }
    Half8(const Half8& lr) {
        value = lr.value;
    }
    Half8(const Half8&& lr) {
        value = std::move(lr.value);
    }

    void set_lane(__fp16 v, const int i) {
        // vsetq_lane_f16(v, value, i);
        value[i] = v;
    }

    const __fp16 operator[](const int i) const {
        // return vgetq_lane_f16(value, i);
        return value[i];
    }

    static Half8 load(const __fp16* addr) {
        Half8 v;
        v.value = vld1q_f16(addr);
        return v;
    }
    static void save(__fp16* addr, const Half8& v) {
        vst1q_f16(addr, v.value);
    }

    static void get_low(const Half8& v1, Half4& v2) {
        v2.value = vget_low_f16(v1.value);
    }
    static void get_high(const Half8& v1, Half4& v2) {
        v2.value = vget_high_f16(v1.value);
    }
    static Half8 combine(const Half4& v1, const Half4& v2) {
        return vcombine_f16(v1.value, v2.value);
    }
    static Half8 extract(const Half8& v1, const Half8& v2, const int n) {
        Half8 dst;
        if (n == 0) {
            dst.value = v1.value;
        } else if (n == 1) {
            dst.value = vextq_f16(v1.value, v2.value, 1);
        } else if (n == 2) {
            dst.value = vextq_f16(v1.value, v2.value, 2);
        } else if (n == 3) {
            dst.value = vextq_f16(v1.value, v2.value, 3);
        } else if (n == 4) {
            dst.value = vextq_f16(v1.value, v2.value, 4);
        } else if (n == 5) {
            dst.value = vextq_f16(v1.value, v2.value, 5);
        } else if (n == 6) {
            dst.value = vextq_f16(v1.value, v2.value, 6);
        } else if (n == 7) {
            dst.value = vextq_f16(v1.value, v2.value, 7);
        } else if (n == 8) {
            dst.value = v2.value;
        }
        return dst;
    }
    static Half8 pad(const Half8& v1, const Half8& v2, const int n) {
        static const uint16_t select  = uint16_t(-1);
        static const uint16x8_t mask1 = {select,select,select,select,select,select,select,0};
        static const uint16x8_t mask2 = {select,select,select,select,select,select,0,0};
        static const uint16x8_t mask3 = {select,select,select,select,select,0,0,0};
        static const uint16x8_t mask4 = {select,select,select,select,0,0,0,0};
        static const uint16x8_t mask5 = {select,select,select,0,0,0,0,0};
        static const uint16x8_t mask6 = {select,select,0,0,0,0,0,0};
        static const uint16x8_t mask7 = {select,0,0,0,0,0,0,0};

        Half8 dst;
        if (n == 0) {
            dst.value = v1.value;
        } else if (n == 1) {
            dst.value = vbslq_f16(mask1, v1.value, v2.value);
        } else if (n == 2) {
            dst.value = vbslq_f16(mask2, v1.value, v2.value);
        } else if (n == 3) {
            dst.value = vbslq_f16(mask3, v1.value, v2.value);
        } else if (n == 4) {
            dst.value = vbslq_f16(mask4, v1.value, v2.value);
        } else if (n == 5) {
            dst.value = vbslq_f16(mask5, v1.value, v2.value);
        } else if (n == 6) {
            dst.value = vbslq_f16(mask6, v1.value, v2.value);
        } else if (n == 7) {
            dst.value = vbslq_f16(mask7, v1.value, v2.value);
        } else if (n == 8) {
            dst.value = v2.value;
        }
        return dst;
    }
    static void mla(Half8& v1, const Half8& v2, const Half8& v3) {
        v1.value = vfmaq_f16(v1.value, v2.value, v3.value);
    }
    static void mla_lane0(Half8& v1, const Half8& v2, const Half4& v3) {
        v1.value = vfmaq_lane_f16(v1.value, v2.value, v3.value, 0);
    }
    static void mla_lane1(Half8& v1, const Half8& v2, const Half4& v3) {
        v1.value = vfmaq_lane_f16(v1.value, v2.value, v3.value, 1);
    }
    static void mla_lane2(Half8& v1, const Half8& v2, const Half4& v3) {
        v1.value = vfmaq_lane_f16(v1.value, v2.value, v3.value, 2);
    }
    static void mla_lane3(Half8& v1, const Half8& v2, const Half4& v3) {
        v1.value = vfmaq_lane_f16(v1.value, v2.value, v3.value, 3);
    }
    static void mls(Half8& v1, const Half8& v2, const Half8& v3) {
        v1.value = vfmsq_f16(v1.value, v2.value, v3.value);
    }
    static void mls_lane0(Half8& v1, const Half8& v2, const Half4& v3) {
        v1.value = vfmsq_lane_f16(v1.value, v2.value, v3.value, 0);
    }
    static void mls_lane1(Half8& v1, const Half8& v2, const Half4& v3) {
        v1.value = vfmsq_lane_f16(v1.value, v2.value, v3.value, 1);
    }
    static void mls_lane2(Half8& v1, const Half8& v2, const Half4& v3) {
        v1.value = vfmsq_lane_f16(v1.value, v2.value, v3.value, 2);
    }
    static void mls_lane3(Half8& v1, const Half8& v2, const Half4& v3) {
        v1.value = vfmsq_lane_f16(v1.value, v2.value, v3.value, 3);
    }
    static Half8 bsl_cle(const Half8& c1, const Half8& c2, const Half8& v1, const Half8& v2) {
        Half8 dst;
        asm volatile (
            "fcmgt %0.8h, %3.8h, %2.8h\n\t"
            "bsl %0.16b, %4.16b, %5.16b\n\t"
            :"=w"(dst.value)
            :"0"(dst.value), "w"(c1.value), "w"(c2.value), "w"(v1.value), "w"(v2.value)
            :"cc", "memory"
        );
        return dst;
    }
    static Half8 bsl_clt(const Half8& c1, const Half8& c2, const Half8& v1, const Half8& v2) {
        Half8 dst;
        asm volatile (
            "fcmge %0.8h, %3.8h, %2.8h\n\t"
            "bsl %0.16b, %4.16b, %5.16b\n\t"
            :"=w"(dst.value)
            :"0"(dst.value), "w"(c1.value), "w"(c2.value), "w"(v1.value), "w"(v2.value)
            :"cc", "memory"
        );
        return dst;
    }
    static Half8 bsl_cge(const Half8& c1, const Half8& c2, const Half8& v1, const Half8& v2) {
        Half8 dst;
        asm volatile (
            "fcmge %0.8h, %2.8h, %3.8h\n\t"
            "bsl %0.16b, %4.16b, %5.16b\n\t"
            :"=w"(dst.value)
            :"0"(dst.value), "w"(c1.value), "w"(c2.value), "w"(v1.value), "w"(v2.value)
            :"cc", "memory"
        );
        return dst;
    }
    static Half8 bsl_cgt(const Half8& c1, const Half8& c2, const Half8& v1, const Half8& v2) {
        Half8 dst;
        asm volatile (
            "fcmgt %0.8h, %2.8h, %3.8h\n\t"
            "bsl %0.16b, %4.16b, %5.16b\n\t"
            :"=w"(dst.value)
            :"0"(dst.value), "w"(c1.value), "w"(c2.value), "w"(v1.value), "w"(v2.value)
            :"cc", "memory"
        );
        return dst;
    }
    static Half8 neg(const Half8& v) {
        Half8 dst;
        dst.value = vnegq_f16(v.value);
        return dst;
    }
    static Half8 floor(const Half8& v) {
        Half8 dst;
        dst.value = vcvtq_f16_s16(vcvtmq_s16_f16(v.value));
        return dst;
    }
    static Half8 ceil(const Half8& v) {
        Half8 dst;
        dst.value = vcvtq_f16_s16(vcvtpq_s16_f16(v.value));
        return dst;
    }
    static Half8 max(const Half8& v1, const Half8& v2) {
        Half8 dst;
        dst.value = vmaxq_f16(v1.value, v2.value);
        return dst;
    }
    static Half8 min(const Half8& v1, const Half8& v2) {
        Half8 dst;
        dst.value = vminq_f16(v1.value, v2.value);
        return dst;
    }
    static Half8 div(const Half8& v1, const Half8& v2) {
        Half8 dst;
        float16x8_t reciprocal = vrecpeq_f16(v2.value);
        reciprocal = vmulq_f16(vrecpsq_f16(v2.value, reciprocal), reciprocal);
        reciprocal = vmulq_f16(vrecpsq_f16(v2.value, reciprocal), reciprocal);
        dst.value = vmulq_f16(v1.value, reciprocal);
        return dst;
    }
    static Half8 exp(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vget_low_f16(v.value));
        float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v.value));
        v_low = exp_ps(v_low);
        v_high = exp_ps(v_high);
        dst.value = vcombine_f16(vcvt_f16_f32(v_low), vcvt_f16_f32(v_high));
        return dst;
    }
    static Half8 pow(const Half8& v, const Half8& e) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vget_low_f16(v.value));
        float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v.value));
        float32x4_t e_low  = vcvt_f32_f16(vget_low_f16(e.value));
        float32x4_t e_high = vcvt_f32_f16(vget_high_f16(e.value));
        v_low = pow_ps(v_low, e_low);
        v_high = pow_ps(v_high, e_high);
        dst.value = vcombine_f16(vcvt_f16_f32(v_low), vcvt_f16_f32(v_high));
        return dst;
    }
    static Half8 sqrt(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vget_low_f16(v.value));
        float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v.value));
        v_low = sqrt_ps(v_low);
        v_high = sqrt_ps(v_high);
        dst.value = vcombine_f16(vcvt_f16_f32(v_low), vcvt_f16_f32(v_high));

        static float16x8_t zero = vdupq_n_f16(0.0f);
        dst.value = vbslq_f16(vceqq_f16(v.value, zero), zero, dst.value);
        return dst;
    }
    static Half8 tanh(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vget_low_f16(v.value));
        float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v.value));
        v_low = tanh_ps(v_low);
        v_high = tanh_ps(v_high);
        dst.value = vcombine_f16(vcvt_f16_f32(v_low), vcvt_f16_f32(v_high));
        return dst;
    }
    static Half8 tan(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vget_low_f16(v.value));
        float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v.value));
        float32x4_t ysin_low, ycos_low;
        float32x4_t ysin_high, ycos_high;
        sincos_ps(v_low, &ysin_low, &ycos_low);
        sincos_ps(v_high, &ysin_high, &ycos_high);
        v_low = div_ps(ysin_low, ycos_low);
        v_high = div_ps(ysin_high, ycos_high);
        dst.value = vcombine_f16(vcvt_f16_f32(v_low), vcvt_f16_f32(v_high));
        return dst;
    }
    static Half8 sin(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vget_low_f16(v.value));
        float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v.value));
        v_low = sin_ps(v_low);
        v_high = sin_ps(v_high);
        dst.value = vcombine_f16(vcvt_f16_f32(v_low), vcvt_f16_f32(v_high));
        return dst;
    }
    static Half8 cos(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vget_low_f16(v.value));
        float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v.value));
        v_low = cos_ps(v_low);
        v_high = cos_ps(v_high);
        dst.value = vcombine_f16(vcvt_f16_f32(v_low), vcvt_f16_f32(v_high));
        return dst;
    }
    static Half8 sigmoid(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vget_low_f16(v.value));
        float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v.value));
        v_low = sigmoid_ps(v_low);
        v_high = sigmoid_ps(v_high);
        dst.value = vcombine_f16(vcvt_f16_f32(v_low), vcvt_f16_f32(v_high));
        return dst;
    }
    static Half8 fast_sigmoid(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vget_low_f16(v.value));
        float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v.value));
        v_low = fast_sigmoid_ps(v_low);
        v_high = fast_sigmoid_ps(v_high);
        dst.value = vcombine_f16(vcvt_f16_f32(v_low), vcvt_f16_f32(v_high));
        return dst;
    }
    static Half8 log(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vget_low_f16(v.value));
        float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v.value));
        v_low = log_ps(v_low);
        v_high = log_ps(v_high);
        dst.value = vcombine_f16(vcvt_f16_f32(v_low), vcvt_f16_f32(v_high));
        return dst;
    }
    static Half8 abs(const Half8& v) {
        Half8 dst;
        dst.value = vabsq_f16(v.value);
        return dst;
    }
    static void zip(Half8& v1, Half8& v2) {
        float16x8x2_t v = vzipq_f16(v1.value, v2.value);
        v1.value = v.val[0];
        v2.value = v.val[1];
    }
    Half8 operator+(const Half8& lr) const {
        Half8 dst;
        dst.value = vaddq_f16(value, lr.value);
        return dst;
    }
    Half8 operator-(const Half8& lr) const {
        Half8 dst;
        dst.value = vsubq_f16(value, lr.value);
        return dst;
    }
    Half8 operator*(__fp16 lr) const {
        Half8 dst;
        dst.value = vmulq_n_f16(value, lr);
        return dst;
    }
    Half8 operator*(const Half8& lr) const {
        Half8 dst;
        dst.value = vmulq_f16(value, lr.value);
        return dst;
    }
    Half8& operator=(const Half8& lr) {
        value = lr.value;
        return *this;
    }
    Half8& operator=(const Half8&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Half8 operator-() const {
        Half8 dst;
        dst.value = -value;
        return dst;
    }
};

struct Half8x4 {
    float16x8x4_t value;
    static Half8x4 ld4(const __fp16* addr) {
        Half8x4 v;
        v.value = vld4q_f16(addr);
        return v;
    }
    void get_lane(Half8& v, int index) {
        v.value = value.val[index];
    }
};

struct Half8x8 {
    float16x8x4_t value0;
    float16x8x4_t value1;
    Half8x8() {}

    void set_value0(const Half8& lr) {
        value0.val[0] = lr.value;
    }
    void set_value1(const Half8& lr) {
        value0.val[1] = lr.value;
    }
    void set_value2(const Half8& lr) {
        value0.val[2] = lr.value;
    }
    void set_value3(const Half8& lr) {
        value0.val[3] = lr.value;
    }
    void set_value4(const Half8& lr) {
        value1.val[0] = lr.value;
    }
    void set_value5(const Half8& lr) {
        value1.val[1] = lr.value;
    }
    void set_value6(const Half8& lr) {
        value1.val[2] = lr.value;
    }
    void set_value7(const Half8& lr) {
        value1.val[3] = lr.value;
    }

    void save_transpose(fp16_t* addr) {
        float16x8x4_t v_tmp0;
        float16x8x4_t v_tmp1;
        v_tmp0.val[0] = vzip1q_f16(value0.val[0], value1.val[0]);
        v_tmp0.val[1] = vzip1q_f16(value0.val[1], value1.val[1]);
        v_tmp0.val[2] = vzip1q_f16(value0.val[2], value1.val[2]);
        v_tmp0.val[3] = vzip1q_f16(value0.val[3], value1.val[3]);
        vst4q_f16(addr, v_tmp0);
        v_tmp1.val[0] = vzip2q_f16(value0.val[0], value1.val[0]);
        v_tmp1.val[1] = vzip2q_f16(value0.val[1], value1.val[1]);
        v_tmp1.val[2] = vzip2q_f16(value0.val[2], value1.val[2]);
        v_tmp1.val[3] = vzip2q_f16(value0.val[3], value1.val[3]);
        vst4q_f16(addr + 32, v_tmp1);
    }
};

#elif defined(TNN_ARM82_A32)

struct Half4 {
    // use int16x4 to store the d register, avoiding compile error 
    int16x4_t value;
    const fp16_t operator[](const int i) const {
        int16_t tmp_v;
        if (i == 0) {
            tmp_v = vget_lane_s16(value, 0);
        } else if (i == 1) {
            tmp_v = vget_lane_s16(value, 1);
        } else if (i == 2) {
            tmp_v = vget_lane_s16(value, 2);
        } else if (i == 3) {
            tmp_v = vget_lane_s16(value, 3);
        }
        return *((fp16_t*)(&tmp_v));
    }
    Half4() {}
    Half4(const int16x4_t& v) {
        value = v;
    }
    Half4(const int16x4_t&& v) {
        value = std::move(v);
    }
    Half4(const Half4& lr) {
        value = lr.value;
    }
    Half4(const Half4&& lr) {
        value = std::move(lr.value);
    }
    Half4(const Float4& lr) {
        value = vreinterpret_s16_f16(vcvt_f16_f32(lr.value));
    }
    static Half4 load(const fp16_t* addr) {
        Half4 v;
        asm volatile(
            "vld1.16 {%P0}, [%2]\n\t"
            :"=w"(v.value)
            :"0"(v.value),"r"(addr)
            :
        );
        return v;
    }
    static void save(fp16_t* addr, const Half4& v) {
        asm volatile(
            "vst1.16 {%P0}, [%1]\n\t"
            :
            :"w"(v.value),"r"(addr)
            :
        );
    }
    static void zip(Half4& v1, Half4& v2) {
        int16x4x2_t v = vzip_s16(v1.value, v2.value);
        v1.value = v.val[0];
        v2.value = v.val[1];
    }
    static void add_to_f32(Half4& v1, Float4& v2) {
        v2.value = vaddq_f32(v2.value, vcvt_f32_f16(vreinterpret_f16_s16(v1.value)));
    }
    Half4& operator=(const Half4& lr) {
        value = lr.value;
        return *this;
    }
};

struct Half8 {
    // use int16x8 to store the q register, avoiding compile error 
    int16x8_t value;
    Half8() {}
    Half8(const fp16_t v) {
        asm volatile(
            "vdup.16 %0, %2\n\t"
            :"=w"(value)
            :"0"(value),"r"(v)
            :
        );
    }
    Half8(const int16x8_t& v) {
        value = v;
    }
    Half8(const int16x8_t&& v) {
        value = std::move(v);
    }
    Half8(const Half8& lr) {
        value = lr.value;
    }
    Half8(const Half8&& lr) {
        value = std::move(lr.value);
    }

    void set_lane(fp16_t v, const int i) {
        fp16_t tmp_v = v;
        if (i == 0) {
            value = vsetq_lane_s16(*((int16_t*)(&tmp_v)), value, 0);
        } else if (i == 1) {
            value = vsetq_lane_s16(*((int16_t*)(&tmp_v)), value, 1);
        } else if (i == 2) {
            value = vsetq_lane_s16(*((int16_t*)(&tmp_v)), value, 2);
        } else if (i == 3) {
            value = vsetq_lane_s16(*((int16_t*)(&tmp_v)), value, 3);
        } else if (i == 4) {
            value = vsetq_lane_s16(*((int16_t*)(&tmp_v)), value, 4);
        } else if (i == 5) {
            value = vsetq_lane_s16(*((int16_t*)(&tmp_v)), value, 5);
        } else if (i == 6) {
            value = vsetq_lane_s16(*((int16_t*)(&tmp_v)), value, 6);
        } else if (i == 7) {
            value = vsetq_lane_s16(*((int16_t*)(&tmp_v)), value, 7);
        }
    }

    void save_lane0(fp16_t* addr) {
        vst1q_lane_s16((int16_t*)addr, value, 0);
    }
    void save_lane1(fp16_t* addr) {
        vst1q_lane_s16((int16_t*)addr, value, 1);
    }
    void save_lane2(fp16_t* addr) {
        vst1q_lane_s16((int16_t*)addr, value, 2);
    }
    void save_lane3(fp16_t* addr) {
        vst1q_lane_s16((int16_t*)addr, value, 3);
    }
    void save_lane4(fp16_t* addr) {
        vst1q_lane_s16((int16_t*)addr, value, 4);
    }
    void save_lane5(fp16_t* addr) {
        vst1q_lane_s16((int16_t*)addr, value, 5);
    }
    void save_lane6(fp16_t* addr) {
        vst1q_lane_s16((int16_t*)addr, value, 6);
    }
    void save_lane7(fp16_t* addr) {
        vst1q_lane_s16((int16_t*)addr, value, 7);
    }

    const fp16_t operator[](const int i) const {
        int16_t tmp_v;
        if (i == 0) {
            tmp_v = vgetq_lane_s16(value, 0);
        } else if (i == 1) {
            tmp_v = vgetq_lane_s16(value, 1);
        } else if (i == 2) {
            tmp_v = vgetq_lane_s16(value, 2);
        } else if (i == 3) {
            tmp_v = vgetq_lane_s16(value, 3);
        } else if (i == 4) {
            tmp_v = vgetq_lane_s16(value, 4);
        } else if (i == 5) {
            tmp_v = vgetq_lane_s16(value, 5);
        } else if (i == 6) {
            tmp_v = vgetq_lane_s16(value, 6);
        } else if (i == 7) {
            tmp_v = vgetq_lane_s16(value, 7);
        }
        return *((fp16_t*)(&tmp_v));
    }

    static Half8 cvt(const uint16x8_t& src) {
        Half8 v;
        asm volatile (
            "vcvtq.f16.u16 %0, %2\n\t"
            :"=w"(v.value)
            :"0"(v.value),"w"(src)
            :
        );
        return v;
    }
    static Half8 load(const fp16_t* addr) {
        Half8 v;
        asm volatile(
            "vld1.16 {%0}, [%2]\n\t"
            :"=w"(v.value)
            :"0"(v.value),"r"(addr)
            :
        );
        return v;
    }
    static void save(fp16_t* addr, const Half8& v) {
        asm volatile(
            "vst1.16 {%0}, [%1]\n\t"
            :
            :"w"(v.value),"r"(addr)
            :
        );
    }
    static void get_low(const Half8& v1, Half4& v2) {
        v2.value = vget_low_s16(v1.value);
    }
    static void get_high(const Half8& v1, Half4& v2) {
        v2.value = vget_high_s16(v1.value);
    }
    static Half8 combine(const Half4& v1, const Half4& v2) {
        return vcombine_s16(v1.value, v2.value);
    }
    static Half8 extract(const Half8& v1, const Half8& v2, const int n) {
        Half8 dst;
        if (n == 0) {
            dst.value = v1.value;
        } else if (n == 1) {
            dst.value = vextq_s16(v1.value, v2.value, 1);
        } else if (n == 2) {
            dst.value = vextq_s16(v1.value, v2.value, 2);
        } else if (n == 3) {
            dst.value = vextq_s16(v1.value, v2.value, 3);
        } else if (n == 4) {
            dst.value = vextq_s16(v1.value, v2.value, 4);
        } else if (n == 5) {
            dst.value = vextq_s16(v1.value, v2.value, 5);
        } else if (n == 6) {
            dst.value = vextq_s16(v1.value, v2.value, 6);
        } else if (n == 7) {
            dst.value = vextq_s16(v1.value, v2.value, 7);
        } else if (n == 8) {
            dst.value = v2.value;
        }
        return dst;
    }
    static Half8 pad(const Half8& v1, const Half8& v2, const int n) {
        static const uint16_t select  = uint16_t(-1);
        static const uint16x8_t mask1 = {select,select,select,select,select,select,select,0};
        static const uint16x8_t mask2 = {select,select,select,select,select,select,0,0};
        static const uint16x8_t mask3 = {select,select,select,select,select,0,0,0};
        static const uint16x8_t mask4 = {select,select,select,select,0,0,0,0};
        static const uint16x8_t mask5 = {select,select,select,0,0,0,0,0};
        static const uint16x8_t mask6 = {select,select,0,0,0,0,0,0};
        static const uint16x8_t mask7 = {select,0,0,0,0,0,0,0};

        Half8 dst;
        if (n == 0) {
            dst.value = v1.value;
        } else if (n == 1) {
            dst.value = vbslq_s16(mask1, v1.value, v2.value);
        } else if (n == 2) {
            dst.value = vbslq_s16(mask2, v1.value, v2.value);
        } else if (n == 3) {
            dst.value = vbslq_s16(mask3, v1.value, v2.value);
        } else if (n == 4) {
            dst.value = vbslq_s16(mask4, v1.value, v2.value);
        } else if (n == 5) {
            dst.value = vbslq_s16(mask5, v1.value, v2.value);
        } else if (n == 6) {
            dst.value = vbslq_s16(mask6, v1.value, v2.value);
        } else if (n == 7) {
            dst.value = vbslq_s16(mask7, v1.value, v2.value);
        } else if (n == 8) {
            dst.value = v2.value;
        }
        return dst;
    }
    static void mla(Half8& v1, const Half8& v2, const Half8& v3) {
        asm volatile (
            "vmla.f16 %0, %2, %3\n\t"
            :"=w"(v1.value)
            :"0"(v1.value),"w"(v2.value),"w"(v3.value)
            :
        );
    }
    static void mla_lane0(Half8& v1, const Half8& v2, const Half4& v3) {
        asm volatile(
            "vmla.f16 %q0, %q2, %P3[0]\n\t"
            :"=w"(v1.value)
            :"0"(v1.value),"w"(v2.value),"w"(v3.value)
            :
        );
    }
    static void mla_lane1(Half8& v1, const Half8& v2, const Half4& v3) {
        asm volatile(
            "vmla.f16 %q0, %q2, %P3[1]\n\t"
            :"=w"(v1.value)
            :"0"(v1.value),"w"(v2.value),"w"(v3.value)
            :
        );
    }
    static void mla_lane2(Half8& v1, const Half8& v2, const Half4& v3) {
        asm volatile(
            "vmla.f16 %q0, %q2, %P3[2]\n\t"
            :"=w"(v1.value)
            :"0"(v1.value),"w"(v2.value),"w"(v3.value)
            :
        );
    }
    static void mla_lane3(Half8& v1, const Half8& v2, const Half4& v3) {
        asm volatile(
            "vmla.f16 %q0, %q2, %P3[3]\n\t"
            :"=w"(v1.value)
            :"0"(v1.value),"w"(v2.value),"w"(v3.value)
            :
        );
    }
    static void mla_3_lanes(Half8& a0, const Half8& m0,
                            Half8& a1, const Half8& m1,
                            Half8& a2, const Half8& m2,
                            const Half4& m) {
        asm volatile(
            "vmov     d0, %9       \n\t"
            "vmla.f16 %0, %6, d0[0]\n\t"
            "vmla.f16 %1, %7, d0[1]\n\t"
            "vmla.f16 %2, %8, d0[2]\n\t"
            :"=w"(a0.value),"=w"(a1.value),"=w"(a2.value)
            :"0"(a0.value),"1"(a1.value),"2"(a2.value),
            "w"(m0.value),"w"(m1.value),"w"(m2.value),"w"(m.value)
            :"cc","q0"
        );
    }
    static void mla_4_lanes(Half8& a0, const Half8& m0,
                            Half8& a1, const Half8& m1,
                            Half8& a2, const Half8& m2,
                            Half8& a3, const Half8& m3,
                            const Half4& m) {
        asm volatile(
            "vmov     d0, %12       \n\t"
            "vmla.f16 %0,  %8, d0[0]\n\t"
            "vmla.f16 %1,  %9, d0[1]\n\t"
            "vmla.f16 %2, %10, d0[2]\n\t"
            "vmla.f16 %3, %11, d0[3]\n\t"
            :"=w"(a0.value),"=w"(a1.value),"=w"(a2.value),"=w"(a3.value)
            :"0"(a0.value),"1"(a1.value),"2"(a2.value),"3"(a3.value),
            "w"(m0.value),"w"(m1.value),"w"(m2.value),"w"(m3.value),"w"(m.value)
            :"cc","q0"
        );
    }
    static void mls(Half8& v1, const Half8& v2, const Half8& v3) {
        asm volatile (
            "vmls.f16 %0, %2, %3\n\t"
            :"=w"(v1.value)
            :"0"(v1.value),"w"(v2.value),"w"(v3.value)
            :
        );
    }
    static void mls_lane0(Half8& v1, const Half8& v2, const Half4& v3) {
        asm volatile(
            "vmls.f16 %q0, %q2, %P3[0]\n\t"
            :"=w"(v1.value)
            :"0"(v1.value),"w"(v2.value),"w"(v3.value)
            :
        );
    }
    static void mls_lane1(Half8& v1, const Half8& v2, const Half4& v3) {
        asm volatile(
            "vmls.f16 %q0, %q2, %P3[1]\n\t"
            :"=w"(v1.value)
            :"0"(v1.value),"w"(v2.value),"w"(v3.value)
            :
        );
    }
    static void mls_lane2(Half8& v1, const Half8& v2, const Half4& v3) {
        asm volatile(
            "vmls.f16 %q0, %q2, %P3[2]\n\t"
            :"=w"(v1.value)
            :"0"(v1.value),"w"(v2.value),"w"(v3.value)
            :
        );
    }
    static void mls_lane3(Half8& v1, const Half8& v2, const Half4& v3) {
        asm volatile(
            "vmls.f16 %q0, %q2, %P3[3]\n\t"
            :"=w"(v1.value)
            :"0"(v1.value),"w"(v2.value),"w"(v3.value)
            :
        );
    }
    static Half8 bsl_cle(const Half8& c1, const Half8& c2, const Half8& v1, const Half8& v2) {
        Half8 dst;
        uint16x8_t cmp_vec;
        asm volatile(
            "vcle.f16 %0, %2, %3\n\t"
            :"=w"(cmp_vec)
            :"0"(cmp_vec),"w"(c1.value),"w"(c2.value)
            :
        );
        dst.value = vbslq_s16(cmp_vec, v1.value, v2.value);
        return dst;
    }
    static Half8 bsl_clt(const Half8& c1, const Half8& c2, const Half8& v1, const Half8& v2) {
        Half8 dst;
        uint16x8_t cmp_vec;
        asm volatile(
            "vclt.f16 %0, %2, %3\n\t"
            :"=w"(cmp_vec)
            :"0"(cmp_vec),"w"(c1.value),"w"(c2.value)
            :
        );
        dst.value = vbslq_s16(cmp_vec, v1.value, v2.value);
        return dst;
    }
    static Half8 bsl_cge(const Half8& c1, const Half8& c2, const Half8& v1, const Half8& v2) {
        Half8 dst;
        uint16x8_t cmp_vec;
        asm volatile(
            "vcge.f16 %0, %2, %3\n\t"
            :"=w"(cmp_vec)
            :"0"(cmp_vec),"w"(c1.value),"w"(c2.value)
            :
        );
        dst.value = vbslq_s16(cmp_vec, v1.value, v2.value);
        return dst;
    }
    static Half8 bsl_cgt(const Half8& c1, const Half8& c2, const Half8& v1, const Half8& v2) {
        Half8 dst;
        uint16x8_t cmp_vec;
        asm volatile(
            "vcgt.f16 %0, %2, %3\n\t"
            :"=w"(cmp_vec)
            :"0"(cmp_vec),"w"(c1.value),"w"(c2.value)
            :
        );
        dst.value = vbslq_s16(cmp_vec, v1.value, v2.value);
        return dst;
    }
    static Half8 neg(const Half8& v) {
        Half8 dst;
        asm volatile(
            "vneg.f16 %0, %2\n\t"
            :"=w"(dst.value)
            :"0"(dst.value),"w"(v.value)
            :
        );
        return dst;
    }
    static Half8 max(const Half8& v1, const Half8& v2) {
        Half8 dst;
        asm volatile(
            "vmax.f16 %0, %2, %3\n\t"
            :"=w"(dst.value)
            :"0"(dst.value),"w"(v1.value),"w"(v2.value)
            :
        );
        return dst;
    }
    static Half8 min(const Half8& v1, const Half8& v2) {
        Half8 dst;
        asm volatile(
            "vmin.f16 %0, %2, %3\n\t"
            :"=w"(dst.value)
            :"0"(dst.value),"w"(v1.value),"w"(v2.value)
            :
        );
        return dst;
    }
    static Half8 div(const Half8& v1, const Half8& v2) {
        Half8 dst;
        asm volatile(
            "vrecpe.f16 q5, %3\n\t"
            "vrecps.f16 q6, %3, q5\n\t"
            "vmul.f16 q5, q6, q5\n\t"
            "vrecps.f16 q6, %3, q5\n\t"
            "vmul.f16 q5, q6, q5\n\t"
            "vmul.f16 %0, %2, q5\n\t"
            :"=w"(dst.value)
            :"0"(dst.value),"w"(v1.value),"w"(v2.value)
            :"q5","q6"
        );
        return dst;
    }
    static Half8 exp(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vreinterpret_f16_s16(vget_low_s16(v.value)));
        float32x4_t v_high = vcvt_f32_f16(vreinterpret_f16_s16(vget_high_s16(v.value)));
        v_low = exp_ps(v_low);
        v_high = exp_ps(v_high);
        dst.value = vcombine_s16(vreinterpret_s16_f16(vcvt_f16_f32(v_low)),
                        vreinterpret_s16_f16(vcvt_f16_f32(v_high)));
        return dst;
    }
    static Half8 pow(const Half8& v, const Half8& e) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vreinterpret_f16_s16(vget_low_s16(v.value)));
        float32x4_t v_high = vcvt_f32_f16(vreinterpret_f16_s16(vget_high_s16(v.value)));
        float32x4_t e_low  = vcvt_f32_f16(vreinterpret_f16_s16(vget_low_s16(e.value)));
        float32x4_t e_high = vcvt_f32_f16(vreinterpret_f16_s16(vget_high_s16(e.value)));
        v_low = pow_ps(v_low, e_low);
        v_high = pow_ps(v_high, e_high);
        dst.value = vcombine_s16(vreinterpret_s16_f16(vcvt_f16_f32(v_low)),
                        vreinterpret_s16_f16(vcvt_f16_f32(v_high)));
        return dst;
    }
    static Half8 sqrt(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vreinterpret_f16_s16(vget_low_s16(v.value)));
        float32x4_t v_high = vcvt_f32_f16(vreinterpret_f16_s16(vget_high_s16(v.value)));
        v_low = sqrt_ps(v_low);
        v_high = sqrt_ps(v_high);
        dst.value = vcombine_s16(vreinterpret_s16_f16(vcvt_f16_f32(v_low)),
                        vreinterpret_s16_f16(vcvt_f16_f32(v_high)));

        uint16x8_t cmp_vec;
        int16x8_t zero = vdupq_n_s16(0);
        asm volatile(
            "vceq.f16 %0, %2, #0\n\t"
            :"=w"(cmp_vec)
            :"0"(cmp_vec),"w"(v.value)
            :
        );
        dst.value = vbslq_s16(cmp_vec, zero, dst.value);
        return dst;
    }
    static Half8 tanh(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vreinterpret_f16_s16(vget_low_s16(v.value)));
        float32x4_t v_high = vcvt_f32_f16(vreinterpret_f16_s16(vget_high_s16(v.value)));
        v_low = tanh_ps(v_low);
        v_high = tanh_ps(v_high);
        dst.value = vcombine_s16(vreinterpret_s16_f16(vcvt_f16_f32(v_low)),
                        vreinterpret_s16_f16(vcvt_f16_f32(v_high)));
        return dst;
    }
    static Half8 tan(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vreinterpret_f16_s16(vget_low_s16(v.value)));
        float32x4_t v_high = vcvt_f32_f16(vreinterpret_f16_s16(vget_high_s16(v.value)));
        float32x4_t ysin_low, ycos_low;
        float32x4_t ysin_high, ycos_high;
        sincos_ps(v_low, &ysin_low, &ycos_low);
        sincos_ps(v_high, &ysin_high, &ycos_high);
        v_low = div_ps(ysin_low, ycos_low);
        v_high = div_ps(ysin_high, ycos_high);
        dst.value = vcombine_s16(vreinterpret_s16_f16(vcvt_f16_f32(v_low)),
                        vreinterpret_s16_f16(vcvt_f16_f32(v_high)));
        return dst;
    }
    static Half8 sin(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vreinterpret_f16_s16(vget_low_s16(v.value)));
        float32x4_t v_high = vcvt_f32_f16(vreinterpret_f16_s16(vget_high_s16(v.value)));
        v_low = sin_ps(v_low);
        v_high = sin_ps(v_high);
        dst.value = vcombine_s16(vreinterpret_s16_f16(vcvt_f16_f32(v_low)),
                        vreinterpret_s16_f16(vcvt_f16_f32(v_high)));
        return dst;
    }
    static Half8 cos(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vreinterpret_f16_s16(vget_low_s16(v.value)));
        float32x4_t v_high = vcvt_f32_f16(vreinterpret_f16_s16(vget_high_s16(v.value)));
        v_low = cos_ps(v_low);
        v_high = cos_ps(v_high);
        dst.value = vcombine_s16(vreinterpret_s16_f16(vcvt_f16_f32(v_low)),
                        vreinterpret_s16_f16(vcvt_f16_f32(v_high)));
        return dst;
    }
    static Half8 sigmoid(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vreinterpret_f16_s16(vget_low_s16(v.value)));
        float32x4_t v_high = vcvt_f32_f16(vreinterpret_f16_s16(vget_high_s16(v.value)));
        v_low = sigmoid_ps(v_low);
        v_high = sigmoid_ps(v_high);
        dst.value = vcombine_s16(vreinterpret_s16_f16(vcvt_f16_f32(v_low)),
                        vreinterpret_s16_f16(vcvt_f16_f32(v_high)));
        return dst;
    }
    static Half8 fast_sigmoid(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vreinterpret_f16_s16(vget_low_s16(v.value)));
        float32x4_t v_high = vcvt_f32_f16(vreinterpret_f16_s16(vget_high_s16(v.value)));
        v_low = fast_sigmoid_ps(v_low);
        v_high = fast_sigmoid_ps(v_high);
        dst.value = vcombine_s16(vreinterpret_s16_f16(vcvt_f16_f32(v_low)),
                        vreinterpret_s16_f16(vcvt_f16_f32(v_high)));
        return dst;
    }
    static Half8 log(const Half8& v) {
        Half8 dst;
        float32x4_t v_low  = vcvt_f32_f16(vreinterpret_f16_s16(vget_low_s16(v.value)));
        float32x4_t v_high = vcvt_f32_f16(vreinterpret_f16_s16(vget_high_s16(v.value)));
        v_low = log_ps(v_low);
        v_high = log_ps(v_high);
        dst.value = vcombine_s16(vreinterpret_s16_f16(vcvt_f16_f32(v_low)),
                        vreinterpret_s16_f16(vcvt_f16_f32(v_high)));
        return dst;
    }
    static Half8 abs(const Half8& v) {
        Half8 dst;
        asm volatile(
            "vabs.f16 %0, %2\n\t"
            :"=w"(dst.value)
            :"0"(dst.value),"w"(v.value)
            :
        );
        return dst;
    }
    static void zip(Half8& v1, Half8& v2) {
        int16x8x2_t v = vzipq_s16(v1.value, v2.value);
        v1.value = v.val[0];
        v2.value = v.val[1];
    }
    Half8 operator+(const Half8& lr) const {
        Half8 dst;
        asm volatile(
            "vadd.f16 %0, %2, %3\n\t"
            :"=w"(dst.value)
            :"0"(dst.value),"w"(value),"w"(lr.value)
            :
        );
        return dst;
    }
    Half8 operator-(const Half8& lr) const {
        Half8 dst;
        asm volatile(
            "vsub.f16 %0, %2, %3\n\t"
            :"=w"(dst.value)
            :"0"(dst.value),"w"(value),"w"(lr.value)
            :
        );
        return dst;
    }
    Half8 operator*(const Half8& lr) const {
        Half8 dst;
        asm volatile(
            "vmul.f16 %0, %2, %3\n\t"
            :"=w"(dst.value)
            :"0"(dst.value),"w"(value),"w"(lr.value)
            :
        );
        return dst;
    }
    Half8& operator=(const Half8& lr) {
        value = lr.value;
        return *this;
    }
    Half8& operator=(const Half8&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Half8 operator-() const {
        Half8 dst;
        asm volatile(
            "vsub.f16 %0, %2\n\t"
            :"=w"(dst.value)
            :"0"(dst.value),"w"(value)
            :
        );
        return dst;
    }
};

struct Half8x4 {
    int16x8x4_t value;
    Half8x4() {}

    static Half8x4 ld4(const fp16_t* addr) {
        Half8x4 v;
        v.value = vld4q_s16((int16_t*)addr);
        return v;
    }
    void get_lane(Half8& v, int index) {
        v.value = value.val[index];
    }

    void set_value0(const Half8& lr) {
        value.val[0] = lr.value;
    }
    void set_value1(const Half8& lr) {
        value.val[1] = lr.value;
    }
    void set_value2(const Half8& lr) {
        value.val[2] = lr.value;
    }
    void set_value3(const Half8& lr) {
        value.val[3] = lr.value;
    }

    const Half8 operator[](const int i) const {
        Half8 tmp_v;
        if (i == 0) {
            tmp_v = Half8(value.val[0]);
        } else if (i == 1) {
            tmp_v = Half8(value.val[1]);
        } else if (i == 2) {
            tmp_v = Half8(value.val[2]);
        } else if (i == 3) {
            tmp_v = Half8(value.val[3]);
        }
        return tmp_v;
    }

    void save_transpose(fp16_t* addr, const Half8& pad) {
        int16x8x4_t v_tmp0;
        int16x8x4_t v_tmp1;
        v_tmp0.val[0] = vzipq_s16(value.val[0], pad.value).val[0];
        v_tmp1.val[0] = vzipq_s16(value.val[0], pad.value).val[1];
        v_tmp0.val[1] = vzipq_s16(value.val[1], pad.value).val[0];
        v_tmp1.val[1] = vzipq_s16(value.val[1], pad.value).val[1];
        v_tmp0.val[2] = vzipq_s16(value.val[2], pad.value).val[0];
        v_tmp1.val[2] = vzipq_s16(value.val[2], pad.value).val[1];
        v_tmp0.val[3] = vzipq_s16(value.val[3], pad.value).val[0];
        v_tmp1.val[3] = vzipq_s16(value.val[3], pad.value).val[1];
        vst4q_s16((int16_t*)addr, v_tmp0);
        vst4q_s16((int16_t*)addr + 32, v_tmp1);
    }
};

struct Half8x8 {
    int16x8x4_t value0;
    int16x8x4_t value1;
    Half8x8() {}

    void set_value0(const Half8& lr) {
        value0.val[0] = lr.value;
    }
    void set_value1(const Half8& lr) {
        value0.val[1] = lr.value;
    }
    void set_value2(const Half8& lr) {
        value0.val[2] = lr.value;
    }
    void set_value3(const Half8& lr) {
        value0.val[3] = lr.value;
    }
    void set_value4(const Half8& lr) {
        value1.val[0] = lr.value;
    }
    void set_value5(const Half8& lr) {
        value1.val[1] = lr.value;
    }
    void set_value6(const Half8& lr) {
        value1.val[2] = lr.value;
    }
    void set_value7(const Half8& lr) {
        value1.val[3] = lr.value;
    }

    void save_transpose(fp16_t* addr) {
        int16x8x4_t v_tmp0;
        int16x8x4_t v_tmp1;
        v_tmp0.val[0] = vzipq_s16(value0.val[0], value1.val[0]).val[0];
        v_tmp1.val[0] = vzipq_s16(value0.val[0], value1.val[0]).val[1];
        v_tmp0.val[1] = vzipq_s16(value0.val[1], value1.val[1]).val[0];
        v_tmp1.val[1] = vzipq_s16(value0.val[1], value1.val[1]).val[1];
        v_tmp0.val[2] = vzipq_s16(value0.val[2], value1.val[2]).val[0];
        v_tmp1.val[2] = vzipq_s16(value0.val[2], value1.val[2]).val[1];
        v_tmp0.val[3] = vzipq_s16(value0.val[3], value1.val[3]).val[0];
        v_tmp1.val[3] = vzipq_s16(value0.val[3], value1.val[3]).val[1];
        vst4q_s16((int16_t*)addr, v_tmp0);
        vst4q_s16((int16_t*)addr + 32, v_tmp1);
    }
};

#else

struct Half4 : TNNVector<fp16_t, 4> {
    using TNNVector<fp16_t, 4>::TNNVector;
    Half4() {}
    Half4(const Half4& lr) {
        for (int i = 0; i < 4; ++i) {
            value[i] = lr.value[i];
        }
    }
    Half4(const Float4& lr) {
        for (int i = 0; i < 4; ++i) {
            value[i] = (fp16_t)lr.value[i];
        }
    }
    Half4(const TNNVector<fp16_t, 4>& lr) {
        for (int i = 0; i < 4; ++i) {
            value[i] = lr.value[i];
        }
    }
    static void add_to_f32(Half4& v1, Float4& v2) {
        for (int i = 0; i < 4; ++i) {
            v2.value[i] = v2.value[i] + (float)v1.value[i];
        }
    }
};

struct Half8 : TNNVector<fp16_t, 8> {
    using TNNVector<fp16_t, 8>::TNNVector;
    Half8() {}
    Half8(const Half8& lr) {
        for (int i = 0; i < 8; ++i) {
            value[i] = lr.value[i];
        }
    }
    Half8(const TNNVector<fp16_t, 8>& lr) {
        for (int i = 0; i < 8; ++i) {
            value[i] = lr.value[i];
        }
    }

    static void get_low(const Half8& v1, Half4& v2) {
        v2.value[0] = v1.value[0];
        v2.value[1] = v1.value[1];
        v2.value[2] = v1.value[2];
        v2.value[3] = v1.value[3];
    }
    static void get_high(const Half8& v1, Half4& v2) {
        v2.value[0] = v1.value[4];
        v2.value[1] = v1.value[5];
        v2.value[2] = v1.value[6];
        v2.value[3] = v1.value[7];
    }
    static Half8 combine(const Half4& v1, const Half4& v2) {
        Half8 dst;
        dst.value[0] = v1.value[0];
        dst.value[1] = v1.value[1];
        dst.value[2] = v1.value[2];
        dst.value[3] = v1.value[3];
        dst.value[4] = v2.value[0];
        dst.value[5] = v2.value[1];
        dst.value[6] = v2.value[2];
        dst.value[7] = v2.value[3];
        return dst;
    }
    static void mlaq_lane0(Half8& v1, const Half8& v2, const Half8& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[0];
        }
    }
    static void mlaq_lane1(Half8& v1, const Half8& v2, const Half8& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[1];
        }
    }
    static void mlaq_lane2(Half8& v1, const Half8& v2, const Half8& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[2];
        }
    }
    static void mlaq_lane3(Half8& v1, const Half8& v2, const Half8& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[3];
        }
    }
    static void mlaq_lane4(Half8& v1, const Half8& v2, const Half8& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[4];
        }
    }
    static void mlaq_lane5(Half8& v1, const Half8& v2, const Half8& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[5];
        }
    }
    static void mlaq_lane6(Half8& v1, const Half8& v2, const Half8& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[6];
        }
    }
    static void mlaq_lane7(Half8& v1, const Half8& v2, const Half8& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[7];
        }
    }
    static void mla_lane0(Half8& v1, const Half8& v2, const Half4& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[0];
        }
    }
    static void mla_lane1(Half8& v1, const Half8& v2, const Half4& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[1];
        }
    }
    static void mla_lane2(Half8& v1, const Half8& v2, const Half4& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[2];
        }
    }
    static void mla_lane3(Half8& v1, const Half8& v2, const Half4& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[3];
        }
    }
    static void mls_lane0(Half8& v1, const Half8& v2, const Half4& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] - v2.value[i] * v3.value[0];
        }
    }
    static void mls_lane1(Half8& v1, const Half8& v2, const Half4& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] - v2.value[i] * v3.value[1];
        }
    }
    static void mls_lane2(Half8& v1, const Half8& v2, const Half4& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] - v2.value[i] * v3.value[2];
        }
    }
    static void mls_lane3(Half8& v1, const Half8& v2, const Half4& v3) {
        for (int i = 0; i < 8; ++i) {
            v1.value[i] = v1.value[i] - v2.value[i] * v3.value[3];
        }
    }
};

#endif

}  // namespace TNN_NS

#endif /* Half8_hpp */
