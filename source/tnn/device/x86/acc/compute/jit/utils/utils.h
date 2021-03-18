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

#ifndef TNN_DEVICE_X86_ACC_COMPUTE_JIT_UTILS_HPP_
#define TNN_DEVICE_X86_ACC_COMPUTE_JIT_UTILS_HPP_


#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include <chrono>
#include <random>
#include <fstream>
#include <exception>

#include <immintrin.h>
#include <xmmintrin.h>

#include "tnn/device/x86/acc/compute/jit/common/type_def.h"

namespace TNN_NS {

inline FILE * tnn_fopen(const char * fname, const char * mode) {
#ifdef _WIN32
    FILE *fp = nullptr;
    return fopen_s(&fp, fname, mode) ? nullptr : fp;
#else
    return fopen(fname, mode);
#endif
}

inline dim_t divDown(dim_t n, dim_t div) {
    return n / div * div;
}

inline dim_t divUp(dim_t n, dim_t div) {
    return (n + div -1 ) / div * div;
}


template <typename T>
int initRandom(T* src, size_t n, T range_min, T range_max) {
    std::mt19937 g(42);
    std::uniform_real_distribution<> rnd(range_min, range_max);

    for (unsigned long long i = 0; i < n; i++) {
        src[i] = static_cast<T>(rnd(g));
    }

    return 0;
}

inline float GFLOPS(int m, int n, int k, float time_ms ) {
    float ops = float(m) * n * k * 2;
    return ops / 1024.0f / 1024.0f / 1024.0f / (time_ms / 1000);
}

inline float DramBW(int m, int n, int k, float time_ms ) {
    float bytes = 0;
    bytes += float(m) * k * sizeof(float);
    bytes += float(n) * k * sizeof(float);
    bytes += float(m) * n * sizeof(float) * 2;
    return bytes / 1024.0f / 1024.0f / 1024.0f / (time_ms / 1000);
}

inline float DramBWPacking(int m, int n, int k, float time_ms ) {
    float bytes = 0;
    bytes += float(m) * k * sizeof(float) * 3; // packing: read + write, compute: read
    bytes += float(n) * k * sizeof(float) * 3;      
    bytes += float(m) * n * sizeof(float) * 2; // computing: read + write
    return bytes / 1024.0f / 1024.0f / 1024.0f / (time_ms / 1000);
}

} // namespace tnn

#endif // TNN_DEVICE_X86_ACC_COMPUTE_JIT_UTILS_HPP_
