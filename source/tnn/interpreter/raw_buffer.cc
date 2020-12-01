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

#include "tnn/interpreter/raw_buffer.h"
#include <fstream>
#include <string>
#include <typeinfo>
#include "tnn/utils/bfp16.h"
#include "tnn/utils/bfp16_utils.h"
#include "tnn/utils/data_type_utils.h"

using namespace TNN_NS;

namespace TNN_NS {
RawBuffer::~RawBuffer() {
    buff_ = nullptr;
}

RawBuffer::RawBuffer() {
    bytes_size_ = 0;
    data_type_  = DATA_TYPE_FLOAT;
}

RawBuffer::RawBuffer(int bytes_size) {
    buff_ = shared_ptr<char>(new char[bytes_size], [](char *p) { delete[] p; });
    memset(buff_.get(), 0, bytes_size);
    bytes_size_ = bytes_size;
}

RawBuffer::RawBuffer(int bytes_size, char *buffer) {
    buff_ = shared_ptr<char>(new char[bytes_size], [](char *p) { delete[] p; });
    memcpy(buff_.get(), buffer, bytes_size);
    bytes_size_ = bytes_size;
}

RawBuffer::RawBuffer(const RawBuffer &buf) {
    this->bytes_size_ = buf.bytes_size_;
    this->data_type_  = buf.data_type_;
    this->buff_       = buf.buff_;
}

template <typename T>
void permute(void *in, void *out, size_t outter, size_t inner) {
    T *in_ptr  = static_cast<T *>(in);
    T *out_ptr = static_cast<T *>(out);
    for (int i = 0; i < outter; i++) {
        for (int j = 0; j < inner; j++) {
            out_ptr[j * outter + i] = in_ptr[i * inner + j];
        }
    }
}

void RawBuffer::Permute(size_t outter, size_t inner) {
    RawBuffer tmp(bytes_size_);
    switch (data_type_) {
        case DATA_TYPE_FLOAT:
            permute<float>(buff_.get(), tmp.buff_.get(), outter, inner);
            break;
        case DATA_TYPE_HALF:
            permute<short>(buff_.get(), tmp.buff_.get(), outter, inner);
            break;
        case DATA_TYPE_INT8:
            permute<int8_t>(buff_.get(), tmp.buff_.get(), outter, inner);
            break;
        default:
            break;
    }

    buff_ = tmp.buff_;
    return;
}

RawBuffer &RawBuffer::operator=(RawBuffer buf) {
    this->bytes_size_ = buf.bytes_size_;
    this->data_type_  = buf.data_type_;
    this->buff_       = buf.buff_;
    return *this;
}

void RawBuffer::buffer(char *buf, int bytes_size) {
    if (bytes_size > bytes_size_) {
        return;
    }
    if (!buff_) {
        buff_ = shared_ptr<char>(new char[bytes_size_], [](char *p) { delete[] p; });
    }
    memcpy(buff_.get(), buf, bytes_size);
    // buff_ = buf;
}

void RawBuffer::SetDataType(DataType data_type) {
    data_type_ = data_type;
}

DataType RawBuffer::GetDataType() {
    return data_type_;
}

int RawBuffer::GetBytesSize() {
    return bytes_size_;
}

int RawBuffer::GetDataCount() {
    int elem_size = DataTypeUtils::GetBytesSize(data_type_);
    return elem_size > 0 ? bytes_size_ / elem_size : 0;
}

/*
 * Convert the data handle form half to Float32
 */
RawBuffer ConvertHalfHandle(RawBuffer &buf) {
    if (buf.GetBytesSize() > 0 && buf.GetDataType() == DATA_TYPE_HALF) {
        auto data_count = buf.GetDataCount();
        RawBuffer buf_f32(data_count * sizeof(float));
        ConvertFromHalfToFloat(buf.force_to<void *>(), buf_f32.force_to<float *>(), data_count);
        return buf_f32;
    } else {
        return buf;
    }
}

/*
 * Convert the data handle form float to bfp16
 */
RawBuffer ConvertFloatToBFP16(RawBuffer &buf) {
    if (buf.GetBytesSize() > 0 && buf.GetDataType() == DATA_TYPE_FLOAT) {
        auto data_count = buf.GetDataCount();
        RawBuffer buf_bfp16(data_count * sizeof(bfp16_t));
        ConvertFromFloatToBFP16(buf.force_to<float *>(), buf_bfp16.force_to<void *>(), data_count);
        buf_bfp16.SetDataType(DATA_TYPE_BFP16);
        return buf_bfp16;
    } else {
        return buf;
    }
}

/*
 * Convert the data handle form half to bfp16
 */
RawBuffer ConvertHalfToBFP16(RawBuffer &buf) {
    if (buf.GetBytesSize() > 0 && buf.GetDataType() == DATA_TYPE_HALF) {
        auto buf_fp32   = ConvertHalfHandle(buf);
        auto data_count = buf_fp32.GetDataCount();
        RawBuffer buf_bfp16(data_count * sizeof(bfp16_t));
        ConvertFromFloatToBFP16(buf_fp32.force_to<float *>(), buf_bfp16.force_to<void *>(), data_count);
        buf_bfp16.SetDataType(DATA_TYPE_BFP16);
        return buf_bfp16;
    } else {
        return buf;
    }
}

std::shared_ptr<float> GetFloatFromRawBuffer(RawBuffer &raw_buffer) {
    int element_size = 0;
    DataType type    = raw_buffer.GetDataType();
    int bytes        = raw_buffer.GetBytesSize();
    if (0 == bytes)
        return nullptr;

    std::shared_ptr<float> float_data;
    if (type == DATA_TYPE_FLOAT) {
        element_size = bytes / sizeof(float);
        float_data.reset(new float[element_size], [](float *p) { delete[] p; });
        memcpy(float_data.get(), raw_buffer.force_to<float *>(), bytes);
    } else if (type == DATA_TYPE_HALF) {
        element_size = bytes / 2;
        float_data.reset(new float[element_size], [](float *p) { delete[] p; });
        ConvertFromHalfToFloat(raw_buffer.force_to<void *>(), float_data.get(), element_size);
    } else if (type == DATA_TYPE_INT8) {
        LOGE("Not support INT8 raw buffer\n");
        return nullptr;
    }

    return float_data;
}

}  // namespace TNN_NS
