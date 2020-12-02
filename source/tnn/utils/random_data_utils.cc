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

#include "tnn/utils/random_data_utils.h"

#include "tnn/utils/bfp16.h"

namespace TNN_NS {

template <typename T>
int InitRandom(T* host_data, size_t n, T range) {
    for (unsigned long long i = 0; i < n; i++) {
        host_data[i] = (T)((rand() % 16 - 8) / 8.0f * range);
    }

    return 0;
}
template int InitRandom(float* host_data, size_t n, float range);
template int InitRandom(int32_t* host_data, size_t n, int32_t range);
template int InitRandom(int8_t* host_data, size_t n, int8_t range);
template int InitRandom(bfp16_t* host_data, size_t n, bfp16_t range);

template <typename T>
int InitRandom(T* host_data, size_t n, T range_min, T range_max) {
    static std::mt19937 g(42);
    std::uniform_real_distribution<> rnd(range_min, range_max);

    for (unsigned long long i = 0; i < n; i++) {
        host_data[i] = static_cast<T>(rnd(g));
    }

    return 0;
}
template int InitRandom(float* host_data, size_t n, float range_min, float range_max);
template int InitRandom(int32_t* host_data, size_t n, int32_t range_min, int32_t range_max);
template int InitRandom(int8_t* host_data, size_t n, int8_t range_min, int8_t range_max);
template int InitRandom(uint8_t* host_data, size_t n, uint8_t range_min, uint8_t range_max);

template <>
int InitRandom(bfp16_t* host_data, size_t n, bfp16_t range_min, bfp16_t range_max) {
    static std::mt19937 g(42);
    std::uniform_real_distribution<> rnd((float)range_min, (float)range_max);

    for (unsigned long long i = 0; i < n; i++) {
        host_data[i] = static_cast<bfp16_t>(rnd(g));
    }

    return 0;
}

}  // namespace TNN_NS
