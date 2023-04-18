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
#include <limits>
#include <vector>

#include "tnn/core/macro.h"
#include "tnn/device/arm/acc/Half8.h"
#include "tnn/interpreter/raw_buffer.h"
namespace TNN_NS {

float RANGE_BOUND = 0.6f;

bool NeedPerChannelQuantize(RawBuffer& raw_buffer, const int channel_size) {
    const int data_count     = raw_buffer.GetDataCount();
    const DataType data_type = raw_buffer.GetDataType();
    ASSERT(data_count % channel_size == 0);
    int stride = data_count / channel_size;
    std::vector<float> min_values(channel_size, std::numeric_limits<float>::max());
    std::vector<float> max_values(channel_size, std::numeric_limits<float>::min());
    if (data_type == DATA_TYPE_FLOAT) {
        float* data = raw_buffer.force_to<float*>();
        for (int i = 0; i < channel_size; ++i) {
            for (int j = 0; j < stride; ++j) {
                float value   = data[i * stride + j];
                min_values[i] = std::min(min_values[i], value);
                max_values[i] = std::max(max_values[i], value);
            }
        }
    } else if (data_type == DATA_TYPE_HALF) {
        fp16_t* data = raw_buffer.force_to<fp16_t*>();
        for (int i = 0; i < channel_size; ++i) {
            for (int j = 0; j < stride; ++j) {
                // fp16_t -> float
                fp16_t value  = data[i * stride + j];
                min_values[i] = std::min(min_values[i], (float)value);
                max_values[i] = std::max(max_values[i], (float)value);
            }
        }
    } else {
        LOGE("NeedPerChannelQuantize does not support data type: %d", data_type);
        return false;
    }
    float sum_range = 0.0f;
    for (int i = 0; i < channel_size; ++i) {
        sum_range += max_values[i] - min_values[i];
    }
    float average_range = sum_range / channel_size;
    if (average_range > RANGE_BOUND) {
        LOGE("The range of weights %.6f overflowed %.6f.\n", average_range, RANGE_BOUND);
        return false;
    }
    return true;
}

bool NeedPerTensorQuantize(RawBuffer& raw_buffer) {
    const int data_count     = raw_buffer.GetDataCount();
    const DataType data_type = raw_buffer.GetDataType();
    float min                = std::numeric_limits<float>::max();
    float max                = std::numeric_limits<float>::min();
    if (data_type == DATA_TYPE_FLOAT) {
        auto data = raw_buffer.force_to<float*>();
        for (int i = 0; i < data_count; ++i) {
            min = std::min(min, data[i]);
            max = std::max(max, data[i]);
        }
    } else if (data_type == DATA_TYPE_HALF) {
        auto data = raw_buffer.force_to<fp16_t*>();
        for (int i = 0; i < data_count; ++i) {
            min = std::min(min, (float)data[i]);
            max = std::max(max, (float)data[i]);
        }
    } else {
        LOGE("NeedPerTensorQuantize does not support data type: %d", data_type);
        return false;
    }
    float sum_range     = max - min;
    float average_range = sum_range / 1;
    if (average_range > RANGE_BOUND) {
        LOGE("The range of weights %.6f overflowed %.6f.\n", average_range, RANGE_BOUND);
        return false;
    }
    return true;
}
}  // namespace TNN_NS
