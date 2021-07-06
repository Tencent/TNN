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

#ifndef TNNCONVERTER_SRC_UTILS_CONVERT_RAW_BUFFER_H
#define TNNCONVERTER_SRC_UTILS_CONVERT_RAW_BUFFER_H
#include "tnn/core/common.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/dims_vector_utils.h"
namespace TNN_CONVERTER {

class ConvertRawBuffer {
public:
    ConvertRawBuffer(ConvertRawBuffer &other) = delete;
    void operator=(const ConvertRawBuffer &) = delete;

    static std::shared_ptr<ConvertRawBuffer> GetInstance();

    TNN_NS::RawBuffer Convert(TNN_NS::RawBuffer &value);
    void SetTargetDataType(TNN_NS::DataType data_type) {
        target_data_type_ = data_type;
    };
    TNN_NS::DataType GetTargetDataType() {
        return target_data_type_;
    }

private:
    ConvertRawBuffer(TNN_NS::DataType data_type) : target_data_type_(data_type){};

    TNN_NS::DataType target_data_type_ = TNN_NS::DATA_TYPE_FLOAT;
    static std::shared_ptr<ConvertRawBuffer> convert_raw_buffer_;
};
}  // namespace TNN_CONVERTER
#endif  // TNNCONVERTER_SRC_UTILS_CONVERT_RAW_BUFFER_H
