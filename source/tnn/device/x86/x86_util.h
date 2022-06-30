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

#ifndef TNN_X86_UTIL_H_
#define TNN_X86_UTIL_H_

#include <string.h>
#include <cstdlib>
#if TNN_PROFILE
#include <chrono>
#endif

#include "tnn/core/blob.h"
#include "tnn/core/macro.h"

namespace TNN_NS {
namespace x86 {
#if TNN_PROFILE
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::time_point;
using std::chrono::system_clock;
struct Timer {
public:
    void Start() {
        start_ = system_clock::now();
    }
    float TimeEclapsed() {
        stop_ = system_clock::now();
        float elapsed = duration_cast<microseconds>(stop_ - start_).count() / 1000.0f;
        start_ = system_clock::now();
        return elapsed;
    }
private:
    time_point<system_clock> start_;
    time_point<system_clock> stop_;
};
#endif

int PackC4(float *dst, const float *src, size_t hw, size_t src_hw_stride, size_t dst_hw_stride, size_t channel);

int PackC8(float *dst, const float *src, size_t hw, size_t src_hw_stride, size_t dst_hw_stride, size_t channel);

int UnpackC4(float *dst, const float *src, size_t hw, size_t src_hw_stride, size_t dst_hw_stride, size_t channel);

int UnpackC8(float *dst, const float *src, size_t hw, size_t src_hw_stride, size_t dst_hw_stride, size_t channel);

template<typename T>
int MatTranspose(T *dst, const T *src, size_t M, size_t N);

int PackINT8Weight(int8_t *src, int8_t *dst, int input_channel, int output_channel, int height, int width);

template<typename T>
T handle_ptr(BlobHandle &handle) {
    return reinterpret_cast<T>(((char*)handle.base) + handle.bytes_offset);
}

template<typename T>
T handle_ptr(BlobHandle &&handle) {
    return reinterpret_cast<T>(((char*)handle.base) + handle.bytes_offset);
}

}  // namespace x86
}  // namespace TNN_NS

#endif
