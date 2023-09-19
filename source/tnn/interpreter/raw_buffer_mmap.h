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

#ifndef TNN_SOURCE_TNN_INTERPRETER_RAW_BUFFER_MMAP_H_
#define TNN_SOURCE_TNN_INTERPRETER_RAW_BUFFER_MMAP_H_


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <memory>
#include "raw_buffer.h"

namespace TNN_NS {

class RawBufferMMap {

public:
    RawBufferMMap();
    RawBufferMMap(RawBuffer& rawBuffer);
    RawBufferMMap(const RawBufferMMap &buf);
    RawBufferMMap &operator=(RawBufferMMap buf);
    ~RawBufferMMap();

    DataType GetDataType() const;
    int GetBytesSize() const;
    int GetDataCount() const;
    DimsVector GetBufferDims() const;

    template <typename T>
    T force_to() {
        return reinterpret_cast<T>(buff_ ? buff_.get() : nullptr);
    }

    template <typename T>
    const T force_to() const {
        return reinterpret_cast<T>(buff_ ? buff_.get() : nullptr);
    }

private:
    std::shared_ptr<char> buff_;

    int bytes_size_ = 0;
    DataType data_type_ = DATA_TYPE_FLOAT;
    DimsVector dims_ = {};
};

} // TNN_NS

#endif //TNN_SOURCE_TNN_INTERPRETER_RAW_BUFFER_MMAP_H_
