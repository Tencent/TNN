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

#include "raw_buffer_mmap.h"

#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

RawBufferMMap::RawBufferMMap() :
  bytes_size_(0),
  data_type_(DATA_TYPE_FLOAT) {}

RawBufferMMap::RawBufferMMap(RawBuffer& rawBuffer) {
    this->bytes_size_ = rawBuffer.GetBytesSize();
    this->data_type_ = rawBuffer.GetDataType();
    this->dims_ = rawBuffer.GetBufferDims();
    char* data = (char*)mmap(NULL, bytes_size_, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (data == MAP_FAILED) {
        LOGE("mmap failed\n");
        this->buff_ = std::shared_ptr<char>(new char[bytes_size_], [](char *p) { delete[] p; });
        memcpy(buff_.get(), rawBuffer.force_to<char *>(), bytes_size_);
    } else {
        memcpy(data, rawBuffer.force_to<char *>(), bytes_size_);
        auto size = this->bytes_size_;
        this->buff_ = std::shared_ptr<char>(data, [=](char *p) {
            munmap(p, size);
        });
    }
}

RawBufferMMap::RawBufferMMap(const RawBufferMMap &buf) {
    this->bytes_size_ = buf.bytes_size_;
    this->data_type_  = buf.data_type_;
    this->buff_       = buf.buff_;
    this->dims_       = buf.dims_;
}

RawBufferMMap &RawBufferMMap::operator=(RawBufferMMap buf)  {
    this->bytes_size_ = buf.bytes_size_;
    this->data_type_  = buf.data_type_;
    this->buff_       = buf.buff_;
    this->dims_       = buf.dims_;
    return *this;
}

RawBufferMMap::~RawBufferMMap() {
    buff_ = nullptr;
}

DataType RawBufferMMap::GetDataType() const {
    return data_type_;
}

int RawBufferMMap::GetBytesSize() const {
    return bytes_size_;
}

int RawBufferMMap::GetDataCount() const {
    int elem_size = DataTypeUtils::GetBytesSize(data_type_);
    return elem_size > 0 ? bytes_size_ / elem_size : 0;
}

DimsVector RawBufferMMap::GetBufferDims() const {
    return dims_;
}

} // TNN_NS