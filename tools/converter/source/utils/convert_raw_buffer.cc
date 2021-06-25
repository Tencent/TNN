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

#include "tools/converter/source/utils/convert_raw_buffer.h"

TNN_NS::DataType ConvertRawBuffer::dst_data_type_ = TNN_NS::DATA_TYPE_FLOAT;

TNN_NS::RawBuffer ConvertRawBuffer::convert(TNN_NS::RawBuffer& value) {
    if (value.GetDataType() == TNN_NS::DATA_TYPE_FLOAT) {
        if (dst_data_type_ == TNN_NS::DATA_TYPE_HALF) {
            return ConvertFloatToHalf(value);
        }
    }

    return value;
}

TNN_NS::RawBuffer ConvertRawBuffer::ConvertFloatToHalf(TNN_NS::RawBuffer& value) {
    const int data_count          = value.GetDataCount();
    TNN_NS::RawBuffer half_buffer = TNN_NS::RawBuffer(data_count * sizeof(fp16_t));
    TNN_NS::ConvertFromFloatToHalf(value.force_to<float*>(), half_buffer.force_to<void*>(), data_count);
    half_buffer.SetDataType(TNN_NS::DATA_TYPE_HALF);

    return half_buffer;
}
