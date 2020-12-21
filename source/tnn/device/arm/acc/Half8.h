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
#include "tnn/utils/half_utils.h"
#include "tnn/device/arm/acc/TNNVector.h"
#ifdef TNN_USE_NEON
#include <arm_neon.h>
#include "tnn/device/arm/acc/neon_mathfun.h"
#endif

namespace TNN_NS {
#if defined(TNN_USE_NEON) && TNN_ARM82 && !defined(TNN_ARM82_SIMU)

struct Half4 {
    float16x4_t value;
    const __fp16 operator[](const int i) const {
        return value[i];
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
    static Half8 bsl_cle(const Half8& c1, const Half8& c2, const Half8& v1, const Half8& v2) {
        Half8 dst;
        dst.value = vbslq_f16(vcleq_f16(c1.value, c2.value), v1.value, v2.value);
        return dst;
    }
    static Half8 bsl_clt(const Half8& c1, const Half8& c2, const Half8& v1, const Half8& v2) {
        Half8 dst;
        asm volatile (
            "fcmlt %0.8h, %2.8h, #0.0\n\t"
            "bsl %0.16b, %3.16b, %4.16b\n\t"
            :"=w"(dst.value)
            :"0"(dst.value), "w"(c1.value), "w"(v1.value), "w"(v2.value)
            :"cc", "memory"
        );
        return dst;
    }
    static Half8 bsl_cge(const Half8& c1, const Half8& c2, const Half8& v1, const Half8& v2) {
        Half8 dst;
        dst.value = vbslq_f16(vcgeq_f16(c1.value, c2.value), v1.value, v2.value);
        return dst;
    }
    static Half8 bsl_cgt(const Half8& c1, const Half8& c2, const Half8& v1, const Half8& v2) {
        Half8 dst;
        dst.value = vbslq_f16(vcgtq_f16(c1.value, c2.value), v1.value, v2.value);
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
    Half8 operator+(const Half8& lr) {
        Half8 dst;
        dst.value = vaddq_f16(value, lr.value);
        return dst;
    }
    Half8 operator-(const Half8& lr) {
        Half8 dst;
        dst.value = vsubq_f16(value, lr.value);
        return dst;
    }
    Half8 operator*(__fp16 lr) {
        Half8 dst;
        dst.value = vmulq_n_f16(value, lr);
        return dst;
    }
    Half8 operator*(const Half8& lr) {
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
    Half8 operator-() {
        Half8 dst;
        dst.value = -value;
        return dst;
    }
};

#else

struct Half4 : TNNVector<fp16_t, 4> {
    using TNNVector<fp16_t, 4>::TNNVector;
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
};

#endif
}  // namespace TNN_NS

#endif /* Half8_hpp */
