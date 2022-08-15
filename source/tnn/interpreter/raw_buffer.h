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

#ifndef TNN_SOURCE_TNN_INTERPRETER_RAW_BUFFER_H_
#define TNN_SOURCE_TNN_INTERPRETER_RAW_BUFFER_H_

#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <typeinfo>
#include "tnn/core/common.h"
#include "tnn/utils/half_utils_inner.h"

namespace TNN_NS {

using std::shared_ptr;

class RawBuffer {
public:
    RawBuffer();
    explicit RawBuffer(int bytes_size);
    RawBuffer(int bytes_size, DimsVector dims);
    RawBuffer(int bytes_size, char *buffer);
    RawBuffer(int bytes_size, char* buffer, DimsVector dims);
    RawBuffer(const RawBuffer &buf);
    RawBuffer(int bytes_size, int alignment);
    RawBuffer &operator=(RawBuffer buf);
    ~RawBuffer();

    void buffer(char *buf, int bytes_size);
    void SetDataType(DataType data_type);
    void SetBufferDims(DimsVector shape);



    DataType GetDataType();
    int GetBytesSize();
    int GetDataCount();
    DimsVector GetBufferDims();

    void Permute(size_t outter, size_t inner);

    template <typename T>
    T force_to() {
        return reinterpret_cast<T>(buff_ ? buff_.get() : nullptr);
    }

private:
    shared_ptr<char> buff_ = nullptr;
    int bytes_size_        = 0;
    DataType data_type_    = DATA_TYPE_FLOAT;
    DimsVector dims_ = {};
};

RawBuffer ConvertFloatToFP16(RawBuffer &buf);
RawBuffer ConvertHalfHandle(RawBuffer &buf);
RawBuffer ConvertFloatToBFP16(RawBuffer &buf);
RawBuffer ConvertHalfToBFP16(RawBuffer &buf);
std::shared_ptr<float> GetFloatFromRawBuffer(RawBuffer &raw_buffer);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_RAW_BUFFER_H_
