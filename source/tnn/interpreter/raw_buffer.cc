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
#include <exception>
#include <fstream>
#include <string>
#include <typeinfo>
#include <utility>
#include "tnn/utils/bfp16.h"
#include "tnn/utils/bfp16_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

using namespace TNN_NS;

namespace TNN_NS {
RawBuffer::~RawBuffer() {
    buff_ = nullptr;
}

RawBuffer::RawBuffer() :
  bytes_size_(0),
  data_type_(DATA_TYPE_FLOAT) {}

RawBuffer::RawBuffer(int bytes_size) {
    if (bytes_size > 0) {
        buff_ = shared_ptr<char>(new char[bytes_size], [](char *p) { delete[] p; });
        memset(buff_.get(), 0, bytes_size);
    } else {
        buff_ = nullptr;
    }

    bytes_size_ = bytes_size;
}

RawBuffer::RawBuffer(int bytes_size, DimsVector dims) : RawBuffer(bytes_size){
    this->dims_ = dims;
}

RawBuffer::RawBuffer(int bytes_size, char *buffer) {
    if (bytes_size > 0) {
        buff_ = shared_ptr<char>(new char[bytes_size], [](char *p) { delete[] p; });
        memcpy(buff_.get(), buffer, bytes_size);
    } else {
        buff_ = nullptr;
    }
    
    bytes_size_ = bytes_size;
}

RawBuffer::RawBuffer(int bytes_size, char* buffer, DimsVector dims) : RawBuffer(bytes_size, buffer) {
          this->dims_ = dims;
}

RawBuffer::RawBuffer(const RawBuffer &buf) {
    this->bytes_size_ = buf.bytes_size_;
    this->data_type_  = buf.data_type_;
    this->buff_       = buf.buff_;
    this->dims_       = buf.dims_;
}

void RawBuffer::SetBufferDims(DimsVector dims) {
    this->dims_ = dims;
}

DimsVector RawBuffer::GetBufferDims() const {
    return this->dims_;
}

void* aligned_malloc(size_t bytes_size, size_t alignment) {
    void* origin_ptr;
    void** align_ptr;
    int offset = alignment - 1 + sizeof(void*);

    origin_ptr = (void*)malloc(bytes_size + offset);
    align_ptr = (void**)(((size_t)(origin_ptr) + offset) & ~(alignment - 1));
    align_ptr[-1] = origin_ptr;
    return align_ptr;
}

void aligned_free(void *align_ptr) {
    free(((void**)align_ptr)[-1]);
}

RawBuffer::RawBuffer(int bytes_size, int alignment) {
    buff_ = shared_ptr<char>(static_cast<char*>(aligned_malloc(bytes_size, alignment)), &aligned_free);
    memset(buff_.get(), 0, bytes_size);
    bytes_size_ = bytes_size;
}

template <typename T>
void permute(void *in, void *out, size_t outter, size_t inner) {
    T *in_ptr  = static_cast<T *>(in);
    T *out_ptr = static_cast<T *>(out);
    for (size_t i = 0; i < outter; i++) {
        for (size_t j = 0; j < inner; j++) {
            out_ptr[j * outter + i] = in_ptr[i * inner + j];
        }
    }
}

void RawBuffer::Reshape(DimsVector& new_dims) {
    //printf("Reshape dims_ %d new_dims: %d \n", DimsVectorUtils::Count(dims_), DimsVectorUtils::Count(new_dims));
    if(DimsVectorUtils::Count(dims_) == DimsVectorUtils::Count(new_dims)) {
        dims_ = new_dims;
    } else {
        throw std::runtime_error("RawBuffer Reshape error \n");
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
    this->dims_       = buf.dims_;
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

DataType RawBuffer::GetDataType() const {
    return data_type_;
}

int RawBuffer::GetBytesSize() const {
    return bytes_size_;
}

int RawBuffer::GetDataCount() const {
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
        buf_f32.SetDataType(DATA_TYPE_FLOAT);
        buf_f32.SetBufferDims(buf.GetBufferDims());
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
        buf_bfp16.SetBufferDims(buf.GetBufferDims());
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
        buf_bfp16.SetBufferDims(buf.GetBufferDims());
        return buf_bfp16;
    } else {
        return buf;
    }
}

RawBuffer ConvertFloatToHalf(RawBuffer &buf) {
    if (buf.GetBytesSize() > 0 && buf.GetDataType() == DATA_TYPE_FLOAT) {
        auto data_count = buf.GetDataCount();
        RawBuffer buf_fp16(data_count * sizeof(fp16_t));
        ConvertFromFloatToHalf(buf.force_to<float *>(), buf_fp16.force_to<void *>(), data_count);
        buf_fp16.SetDataType(DATA_TYPE_HALF);
        buf_fp16.SetBufferDims(buf.GetBufferDims());
        return buf_fp16;
    } else {
        return buf;
    }
}

std::shared_ptr<float> GetFloatFromRawBuffer(const RawBuffer &raw_buffer) {
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

RawBuffer ConvertFloatToFP16(RawBuffer &buf) {
    if (buf.GetBytesSize() > 0 && buf.GetDataType() == DATA_TYPE_FLOAT) {
        int data_count = buf.GetDataCount();
        RawBuffer buf_fp16(data_count * sizeof(fp16_t));
        buf_fp16.SetDataType(DATA_TYPE_HALF);
        buf_fp16.SetBufferDims(buf.GetBufferDims());
        ConvertFromFloatToHalf(buf.force_to<float *>(), buf_fp16.force_to<fp16_t *>(), data_count);
        return buf_fp16;
    } else {
        return buf;
    }
}

RawBuffer Concat(std::vector<RawBuffer> & list, int axis) {
    RawBuffer buffer0 = list[0];
    auto buffer0_dims = buffer0.GetBufferDims();
    for(int i= 1; i < list.size(); ++i) {
        auto dims = list[i].GetBufferDims();
        if(buffer0_dims.size() != dims.size() && axis >= buffer0_dims.size()) {
             throw std::runtime_error("RawBuffer Concat Error \n");
        }
    }
    auto output_dims = buffer0_dims;
    int out_concat_dim_size = 0;
    for(int i = 0; i <list.size(); ++i) {
        out_concat_dim_size += list[i].GetBufferDims()[axis];
    }
    output_dims[axis] = out_concat_dim_size;
    int num_concats = DimsVectorUtils::Count(buffer0_dims, 0, axis);
    auto datasize                 = DataTypeUtils::GetBytesSize(buffer0.GetDataType());

    RawBuffer output_buffer(DimsVectorUtils::Count(output_dims) * datasize);
    output_buffer.SetBufferDims(output_dims);
    int8_t* output_data = output_buffer.force_to<int8_t*>();

    int concate_size = DimsVectorUtils::Count(buffer0_dims, axis+ 1);
    int output_concat_axis_offset = 0;

    for (size_t i = 0; i < list.size(); ++i) {
        int8_t *input_data          = list[i].force_to<int8_t*>();
        const int input_concat_axis = list[i].GetBufferDims()[axis];
        for (int n = 0; n < num_concats; ++n) {
            memcpy(output_data + (n * out_concat_dim_size + output_concat_axis_offset) * concate_size * datasize,
                   input_data + n * input_concat_axis * concate_size * datasize,
                   input_concat_axis * concate_size * datasize);
        }
        output_concat_axis_offset += input_concat_axis;
    }
    return output_buffer;
}

}  // namespace TNN_NS
