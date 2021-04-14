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

#ifndef TNNVector_hpp
#define TNNVector_hpp
#include <algorithm>  // supply std::max and std::min
#include <cmath>
#include "tnn/core/macro.h"

namespace TNN_NS {

template <typename T, unsigned int len>
struct TNNVector {
    T value[len];
    TNNVector<T, len> operator+(const TNNVector<T, len>& lr) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = value[i] + lr.value[i];
        }
        return dst;
    }
    TNNVector<T, len> operator-(const TNNVector<T, len>& lr) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = value[i] - lr.value[i];
        }
        return dst;
    }
    TNNVector<T, len> operator*(const TNNVector<T, len>& lr) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = value[i] * lr.value[i];
        }
        return dst;
    }
    TNNVector<T, len> operator*(T lr) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = value[i] * lr;
        }
        return dst;
    }

    TNNVector<T, len>& operator=(const TNNVector<T, len>& lr) {
        for (int i = 0; i < len; ++i) {
            value[i] = lr.value[i];
        }
        return *this;
    }
    TNNVector<T, len> operator-() {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = -value[i];
        }
        return dst;
    }
    TNNVector<T, len>() {}
    TNNVector<T, len>(const T v) {
        for (int i = 0; i < len; ++i) {
            value[i] = v;
        }
    }

    TNNVector<T, len>(const TNNVector<T, len>& lr) {
        for (int i = 0; i < len; ++i) {
            value[i] = lr.value[i];
        }
    }

    TNNVector<T, len>(const float *addr) {
        for (int i = 0; i < len; ++i) {
            value[i] = *addr;
        }
    }

    void set_lane(T v, int i) {
        value[i] = v;
    }

    const T operator[](const int i) const {
        return value[i];
    }

    static TNNVector<T, len> load(const T* addr) {
        TNNVector<T, len> v;
        for (int i = 0; i < len; ++i) {
            v.value[i] = addr[i];
        }
        return v;
    }
    static TNNVector<T, len> loadu(const T* addr) {
        return load(addr);
    }
    static void save(T* addr, const TNNVector<T, len>& v) {
        for (int i = 0; i < len; ++i) {
            addr[i] = v.value[i];
        }
    }
    static void saveu(T* addr, const TNNVector<T, len>& v) {
        save(addr, v);
    }

    static TNNVector<T, len> bsl_cle(const TNNVector<T, len>& c1, const TNNVector<T, len>& c2, const TNNVector<T, len>& v1, const TNNVector<T, len>& v2) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = c1.value[i] <= c2.value[i] ? v1.value[i] : v2.value[i];
        }
        return dst;
    }
    static TNNVector<T, len> bsl_clt(const TNNVector<T, len>& c1, const TNNVector<T, len>& c2, const TNNVector<T, len>& v1, const TNNVector<T, len>& v2) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = c1.value[i] < c2.value[i] ? v1.value[i] : v2.value[i];
        }
        return dst;
    }
    static TNNVector<T, len> bsl_cge(const TNNVector<T, len>& c1, const TNNVector<T, len>& c2, const TNNVector<T, len>& v1, const TNNVector<T, len>& v2) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = c1.value[i] >= c2.value[i] ? v1.value[i] : v2.value[i];
        }
        return dst;
    }
    static TNNVector<T, len> bsl_cgt(const TNNVector<T, len>& c1, const TNNVector<T, len>& c2, const TNNVector<T, len>& v1, const TNNVector<T, len>& v2) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = c1.value[i] > c2.value[i] ? v1.value[i] : v2.value[i];
        }
        return dst;
    }

    static TNNVector<T, len> extract(const TNNVector<T, len>& v1, const TNNVector<T, len>& v2, const int n) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = (n + i < len) ? v1[n + i] : v2[n + i - len];
        }
        return dst;
    }
    static TNNVector<T, len> pad(const TNNVector<T, len>& v1, const TNNVector<T, len>& v2, const int n) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len - n; ++i) {
            dst.value[i] = v1[i];
        }
        for (int i = len - n; i < len; ++i) {
            dst.value[i] = v2[i];
        }
        return dst;
    }
    static void mla(TNNVector<T, len>& v1, const TNNVector<T, len>& v2, const TNNVector<T, len>& v3) {
        for (int i = 0; i < len; ++i) {
            v1.value[i] = v1.value[i] + v2.value[i] * v3.value[i];
        }
    }
    static void mls(TNNVector<T, len>& v1, const TNNVector<T, len>& v2, const TNNVector<T, len>& v3) {
        for (int i = 0; i < len; ++i) {
            v1.value[i] = v1.value[i] - v2.value[i] * v3.value[i];
        }
    }
    static TNNVector<T, len> neg(const TNNVector<T, len>& v) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = -v.value[i];
        }
        return dst;
    }
    static TNNVector<T, len> max(const TNNVector<T, len>& v1, const TNNVector<T, len>& v2) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = std::max(v1.value[i], v2.value[i]);
        }
        return dst;
    }
    static TNNVector<T, len> min(const TNNVector<T, len>& v1, const TNNVector<T, len>& v2) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = std::min(v1.value[i], v2.value[i]);
        }
        return dst;
    }
    static TNNVector<T, len> div(const TNNVector<T, len>& v1, const TNNVector<T, len>& v2) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = v1.value[i] / v2.value[i];
        }
        return dst;
    }
    static TNNVector<T, len> add(const TNNVector<T, len>& v1, const TNNVector<T, len>& v2) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = v1.value[i] + v2.value[i];
        }
        return dst;
    }
    static TNNVector<T, len> sub(const TNNVector<T, len>& v1, const TNNVector<T, len>& v2) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = v1.value[i] - v2.value[i];
        }
        return dst;
    }
    static TNNVector<T, len> mul(const TNNVector<T, len>& v1, const TNNVector<T, len>& v2) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = v1.value[i] * v2.value[i];
        }
        return dst;
    }

    // the following functions only work for fp32 and fp16, int8 need to override these functions
    static TNNVector<T, len> floor(const TNNVector<T, len>& v) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = std::floor(v.value[i]);
        }
        return dst;
    }
    static TNNVector<T, len> ceil(const TNNVector<T, len>& v) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = std::ceil(v.value[i]);
        }
        return dst;
    }
    static TNNVector<T, len> exp(const TNNVector<T, len>& v) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = std::exp(v.value[i]);
        }
        return dst;
    }
    static TNNVector<T, len> pow(const TNNVector<T, len>& v, const TNNVector<T, len>& e) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            if (v.value[i] <= 0) {
                LOGE("%s\n", "neon pow does not support zero or negative input value");
            }
            dst.value[i] = std::pow(v.value[i], e.value[i]);
        }
        return dst;
    }
    static TNNVector<T, len> sqrt(const TNNVector<T, len>& v) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = std::sqrt(v.value[i]);
        }
        return dst;
    }
    static TNNVector<T, len> tanh(const TNNVector<T, len>& v) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = std::tanh(v.value[i]);
        }
        return dst;
    }

    static TNNVector<T, len> tan(const TNNVector<T, len>& v) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = std::tan(v.value[i]);
        }
        return dst;
    }
    static TNNVector<T, len> sin(const TNNVector<T, len>& v) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = std::sin(v.value[i]);
        }
        return dst;
    }
    static TNNVector<T, len> cos(const TNNVector<T, len>& v) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = std::cos(v.value[i]);
        }
        return dst;
    }
    static TNNVector<T, len> sigmoid(const TNNVector<T, len>& v) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = 1.0f / (1.0f + std::exp(-v.value[i]));
        }
        return dst;
    }
    static TNNVector<T, len> fast_sigmoid(const TNNVector<T, len>& v) {
        return TNNVector<T, len>::sigmoid(v);
    }
    static TNNVector<T, len> log(const TNNVector<T, len>& v) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = std::log(v.value[i]);
        }
        return dst;
    }
    static TNNVector<T, len> abs(const TNNVector<T, len>& v) {
        TNNVector<T, len> dst;
        for (int i = 0; i < len; ++i) {
            dst.value[i] = std::fabs(v.value[i]);
        }
        return dst;
    }
    static void zip(TNNVector<T, len>& v1, TNNVector<T, len>& v2) {
        if (len % 2 != 0) {
            LOGE("%s\n", "vecotr zip does not support len is odd");
        }
        T tmp[len];
        for (int i = 0; i < len; i++) {
            tmp[i] = v1.value[i];
        }
        for (int i = 0; i < len / 2; i++) {
            v1.value[i * 2] = tmp[i];
            v1.value[i * 2 + 1] = v2.value[i];
        }
        for (int i = 0; i < len / 2; i++) {
            v2.value[i * 2] = tmp[len / 2 + i];
            v2.value[i * 2 + 1] = v2.value[len / 2 + i];
        }
    }
};

}  // namespace TNN_NS

#endif /* TNNVector_hpp */
