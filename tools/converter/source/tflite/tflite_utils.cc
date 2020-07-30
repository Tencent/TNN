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

#include "tflite_utils.h"

#include "tnn/core/macro.h"

namespace TNN_CONVERTER {

bool ConvertDataFormatTFLite(const float* src, float* dst, int KH, int KW, int CI, int CO) {
    ASSERT(KH > 0);
    ASSERT(KW > 0);
    ASSERT(CI > 0);
    ASSERT(CO > 0);
    ASSERT(src != nullptr);
    // CO KH KW CI --> CO CI KH KW
    for (int oc = 0; oc < CO; ++oc) {
        for (int ic = 0; ic < CI; ++ic) {
            for (int h = 0; h < KH; ++h) {
                for (int w = 0; w < KW; ++w) {
                    dst[(oc * CI + ic) * KH * KW + h * KW + w] = src[(oc * KH + h) * KW * CI + w * CI + ic];
                }
            }
        }
    }
    return true;
}

bool ConvertShapeFormatTFLite(std::vector<int32_t>& shape) {
    // shape [n, h , w, c] -> shape [n, c, h, w]
    if (shape.size() == 4) {
        auto h   = shape[1];
        auto w   = shape[2];
        auto c   = shape[3];
        shape[1] = c;
        shape[2] = h;
        shape[3] = w;
        return true;
    } else if (shape.size() < 4 && shape.size() > 0) {
        int shape_size = shape.size();
        for (int i = 0; i < 4 - shape_size; i++) {
            shape.push_back(1);
        }
        return true;
    } else {
        LOGE("TNN Converter do not support wrong shape!\n");
        return false;
    }
}

// template <typename T>
bool ConvertConstFormatTFLite(int32_t const* dst, int32_t const* src, std::vector<int32_t> shape) {
    ASSERT(shape.size() == 2);
    ASSERT(shape[0] == 4);
    int data_size = shape[1];
    // std::memcpy((void*)(dst + 0 * data_size), src + 0 * data_size, data_size*sizeof(int32_t));
    std::memcpy((void*)(dst + 0 * data_size), src + 2 * data_size, data_size * sizeof(int32_t));
    std::memcpy((void*)(dst + 1 * data_size), src + 1 * data_size, data_size * sizeof(int32_t));
    std::memcpy((void*)(dst + 2 * data_size), src + 3 * data_size, data_size * sizeof(int32_t));
    return true;
}

int ConvertAxisFormatTFLite(int axis) {
    assert(axis > -4 && axis < 4);
    if (axis < 0) {
        axis += 4;
    }
    switch (axis) {
        case 0:
            return 0;
        case 1:
            return 2;
        case 2:
            return 3;
        default:
            return 1;
    }
}

int Count(std::vector<int> shape) {
    if (shape.empty()) {
        return 0;
    }
    int count = 1;
    for (auto i : shape) {
        count *= i;
    }
    return count;
}
}  // namespace TNN_CONVERTER