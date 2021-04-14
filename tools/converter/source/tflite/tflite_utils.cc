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

#include <cstring>

#include "tnn/core/macro.h"

namespace TNN_CONVERTER {

bool TFLiteConvertOHWI2OIHW(const float* src, float* dst, int CO, int KH, int KW, int CI) {
    ASSERT(CO > 0);
    ASSERT(KH > 0);
    ASSERT(KW > 0);
    ASSERT(CI > 0);
    ASSERT(src != nullptr);
    for (int co = 0; co < CO; ++co) {
        for (int ci = 0; ci < CI; ++ci) {
            for (int h = 0; h < KH; ++h) {
                for (int w = 0; w < KW; ++w) {
                    dst[(co * CI + ci) * KH * KW + h * KW + w] = src[(co * KH + h) * KW * CI + w * CI + ci];
                }
            }
        }
    }
    return true;
}

bool TFLiteConvertOHWI2IOHW(const float* src, float* dst, int CO, int KH, int KW, int CI) {
    ASSERT(CI > 0);
    ASSERT(KH > 0);
    ASSERT(KW > 0);
    ASSERT(CO > 0);
    ASSERT(src != nullptr);
    for (int ci = 0; ci < CI; ++ci) {
        for (int co = 0; co < CO; ++co) {
            for (int h = 0; h < KH; ++h) {
                for (int w = 0; w < KW; ++w) {
                    dst[(ci * CO + co) * KH * KW + h * KW + w] = src[(co * KH + h) * KW * CI + w * CI + ci];
                }
            }
        }
    }
    return true;
}

bool ConvertShapeFormatTFLite(std::vector<int32_t>& shape) {
    if (shape.empty()) {
        LOGE("TNN Converter do not support wrong shape!\n");
        return false;
    }
    while (shape.size() < 4) {
        shape.insert(shape.end() - 1, 1);
    }
    // shape [n, h , w, c] -> shape [n, c, h, w]
    if (shape.size() == 4) {
        auto h   = shape[1];
        auto w   = shape[2];
        auto c   = shape[3];
        shape[1] = c;
        shape[2] = h;
        shape[3] = w;
    }
    return true;
}

bool ConvertPermFormatTFLite(std::vector<int32_t>& perm) {
    if (perm.empty()) {
        LOGE("TNN Converter do not support wrong perm!\n");
        return false;
    }

    int perm_size = perm.size();
    if (perm_size > 4) {
        LOGE("TNN Transpose do not support perm's size larger than 4!\n");
        return false;
    }

    for (int i = perm_size; i < 4; i++) {
        perm.emplace_back(i);
    }

    std::map<int, int> nhwc_to_nchw;
    nhwc_to_nchw[0] = 0;
    nhwc_to_nchw[1] = 2;
    nhwc_to_nchw[2] = 3;
    nhwc_to_nchw[3] = 1;

    for (auto& v : perm) {
        v = nhwc_to_nchw[v];
    }
    ConvertShapeFormatTFLite(perm);

    return true;
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

int ConvertAxisFormatTFLite(int axis, int input_shape_size) {
    assert(axis > -4 && axis < 4);
    if (axis < 0) {
        axis += input_shape_size;
    }

    if (input_shape_size == 2) {
        return axis;
    } else if (input_shape_size == 3) {
        switch (axis) {
            case 1:
                return 2;
            case 2:
                return 1;
            default:
                return 0;
        }
    } else if (input_shape_size == 4) {
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

    return axis;
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

int SizeofTFLiteTensorData(tflite::TensorType type) {
    switch (type) {
        case tflite::TensorType_FLOAT32:
            return sizeof(float);
        case tflite::TensorType_INT32:
            return sizeof(int32_t);
        case tflite::TensorType_INT16:
            return sizeof(int16_t);
        case tflite::TensorType_INT64:
            return sizeof(int64_t);
        default:
            return 0;
    }
    return 0;
}

void Mask(std::vector<int> shape, int mask, int upper, std::vector<int>& v) {
    ASSERT(shape.size() == 4);
    ASSERT(v.size() == 4);
    ASSERT(mask <= 15 && mask >= 0);
    if (upper == 0) {
        // 处理的是 begin，取的是 0
        if (mask & 0x1)
            v[0] = 0;
        if (mask & 0x2)
            v[1] = 0;
        if (mask & 0x4)
            v[2] = 0;
        if (mask & 0x8)
            v[3] = 0;
    } else {
        // 处理的是 ends， 取最大值
        if (mask & 0x1)
            v[0] = shape[0];
        if (mask & 0x2)
            v[1] = shape[1];
        if (mask & 0x4)
            v[2] = shape[2];
        if (mask & 0x8)
            v[3] = shape[3];
    }
}
}  // namespace TNN_CONVERTER
