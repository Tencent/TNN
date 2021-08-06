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

#include "convert_raw_buffer.h"

namespace TNN_CONVERTER {

std::shared_ptr<ConvertRawBuffer> ConvertRawBuffer::convert_raw_buffer_ = nullptr;

std::shared_ptr<ConvertRawBuffer> ConvertRawBuffer::GetInstance() {
    if (convert_raw_buffer_ == nullptr) {
        convert_raw_buffer_ = std::shared_ptr<ConvertRawBuffer>(new ConvertRawBuffer(TNN_NS::DATA_TYPE_FLOAT));
    }
    return convert_raw_buffer_;
}

TNN_NS::RawBuffer ConvertRawBuffer::Convert(TNN_NS::RawBuffer &value) {
    if (value.GetDataType() == TNN_NS::DATA_TYPE_FLOAT &&
        target_data_type_ == TNN_NS::DATA_TYPE_HALF) {
        const int data_count          = value.GetDataCount();
        TNN_NS::RawBuffer half_buffer = TNN_NS::RawBuffer(data_count * sizeof(fp16_t));
        TNN_NS::ConvertFromFloatToHalf(value.force_to<float*>(), half_buffer.force_to<void*>(), data_count);
        half_buffer.SetDataType(TNN_NS::DATA_TYPE_HALF);
        return half_buffer;
    } else if (value.GetDataType() ==  target_data_type_ ) {
        return value;
    } else {
        LOGE("ConvertRawBuffer does not support convert from %d to %d", value.GetDataType(), target_data_type_);
        return value;
    }
}
}  // namespace TNN_CONVERTER
