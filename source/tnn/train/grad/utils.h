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

// author: sanerzheng@tencent.com

#ifndef TNN_SOURCE_TNN_TRAIN_GRAD_UTILS_H
#define TNN_SOURCE_TNN_TRAIN_GRAD_UTILS_H

#include <vector>

#include "tnn/core/blob.h"
#include "tnn/core/status.h"
#include "tnn/layer/base_layer.h"
#include "tnn/train/operations/op_type.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {
class AbstractNetwork;
namespace train {
#define DECLARE_PARAMWRAPPER_FUNCS(TypeName, ValueName)                                                                \
    ParamWrapper(TypeName v) {                                                                                         \
        value_.ValueName = v;                                                                                          \
        type_            = ParamType::ValueName##_enum;                                                                \
    };                                                                                                                 \
    inline TypeName Get##TypeName() {                                                                                  \
        assert(type_ == ParamType::ValueName##_enum);                                                                  \
        return value_.ValueName;                                                                                       \
    };                                                                                                                 \
    inline bool Is##TypeName() {                                                                                       \
        return type_ == ParamType::ValueName##_enum;                                                                   \
    };

#define DECLARE_PARAMWRAPPER_POINTER_FUNCS(TypeName, ValueName)                                                        \
    ParamWrapper(TypeName *v) {                                                                                        \
        value_.ValueName = v;                                                                                          \
        type_            = ParamType::ValueName##_enum;                                                                \
    };                                                                                                                 \
    inline TypeName *Get##TypeName##Pointer() {                                                                        \
        assert(type_ == ParamType::ValueName##_enum);                                                                  \
        return value_.ValueName;                                                                                       \
    };                                                                                                                 \
    inline bool Is##TypeName##Pointer() {                                                                              \
        return type_ == ParamType::ValueName##_enum;                                                                   \
    };

int ConvertFromFloatToBFP16(float *fp32, void *fp16, int count);
int ConvertFromBFP16ToFloat(void *fp16, float *fp32, int count);
void ConvertToNCHW(void *&src_ptr, RawBuffer &dst, RawBuffer *input_rawbuffer);
void ConvertToNCHW(void *&data_ptr, RawBuffer &dst, const BlobDesc &input_desc);
void ConvertToNCHW(void *&src_ptr, RawBuffer &dst, const DataType &dtype, const DataFormat &dformat,
                   const DimsVector &dims);
void ConvertToNC4HW4(std::shared_ptr<RawBuffer> &src, BlobDesc &input_desc);
int CalculateElementCount(const BlobDesc &desc);
int CalculateElementCount(const DataFormat data_format, const DimsVector &dims, const DataType data_type);
void ConvertToNCHW(std::shared_ptr<RawBuffer> src, std::shared_ptr<RawBuffer>& dst);
inline void* GetBlobHandle(Blob *blob) {
    return static_cast<void *>(static_cast<char *>(blob->GetHandle().base) + blob->GetHandle().bytes_offset);
}

class ParamWrapper {
public:
    typedef std::map<Blob *, std::shared_ptr<RawBuffer>> Blob2RawBufferMap;
    typedef std::map<Blob *, std::shared_ptr<Blob>> Blob2BlobMap;
    // typedef std::shared_ptr<Blob> BlobSharedPtr;
    // typedef std::shared_ptr<RawBuffer> RawBufferSharedPtr;
    // @brief ParamType is used for safe, put type must same with get type
    enum ParamType {
        default_type_enum = 0,
        int_value_enum,
        bool_value_enum,
        long_value_enum,
        float_value_enum,
        bfp16_value_enum,
        blob_pvalue_enum,
        element_op_value_enum,
        blob_shared_ptr_value_enum,
        rawbuffer_shared_ptr_value_enum,
        base_layer_pvalue_enum,
        layer_resource_pvalue_enum,
        network_pvalue_enum,
        layer_param_pvalue_enum,
        blob_2_blob_pvalue_enum,
        blob_raw_buffer_pvalue_enum,
        raw_buffer_pvalue_enum,
        void_pvalue_enum
    };
    union ParamValue {
        int int_value;
        bool bool_value;
        long long_value;
        float float_value;
        ElementOpType element_op_value;
        Blob *blob_pvalue;
        BaseLayer *base_layer_pvalue;
        LayerResource *layer_resource_pvalue;
        AbstractNetwork *network_pvalue;
        LayerParam *layer_param_pvalue;
        Blob2BlobMap *blob_2_blob_pvalue;
        Blob2RawBufferMap *blob_raw_buffer_pvalue;
        RawBuffer *raw_buffer_pvalue;
        void *void_pvalue;
        // BlobSharedPtr blob_shared_ptr_value;
        // RawBufferSharedPtr rawbuffer_shared_ptr_value;
        // ParamValue(){};
        // ~ParamValue(){};
    };
    ParamWrapper() : type_(ParamType::default_type_enum){};
    ParamWrapper(const ParamWrapper &other) {
        if (this != &other) {
            type_                  = other.type_;
            value_                 = other.value_;
            blob_shared_ptr_       = other.blob_shared_ptr_;
            raw_buffer_shared_ptr_ = other.raw_buffer_shared_ptr_;
            bfp16_value_           = other.bfp16_value_;
        }
    };
    ParamWrapper(const ParamWrapper &&other) {
        if (this != &other) {
            type_                  = other.type_;
            value_                 = other.value_;
            blob_shared_ptr_       = std::move(other.blob_shared_ptr_);
            raw_buffer_shared_ptr_ = std::move(other.raw_buffer_shared_ptr_);
            bfp16_value_           = other.bfp16_value_;
        }
    };
    ParamWrapper &operator=(const ParamWrapper &other) {
        if (this != &other) {
            type_                  = other.type_;
            value_                 = other.value_;
            blob_shared_ptr_       = other.blob_shared_ptr_;
            raw_buffer_shared_ptr_ = other.raw_buffer_shared_ptr_;
            bfp16_value_           = other.bfp16_value_;
        }
        return *this;
    };
    ParamWrapper &operator=(const ParamWrapper &&other) {
        if (this != &other) {
            type_                  = other.type_;
            value_                 = other.value_;
            blob_shared_ptr_       = std::move(other.blob_shared_ptr_);
            raw_buffer_shared_ptr_ = std::move(other.raw_buffer_shared_ptr_);
            bfp16_value_           = other.bfp16_value_;
        }
        return *this;
    };
    ~ParamWrapper(){
        // if(type_ == ParamType::blob_shared_ptr_value_enum) {
        //     //TODO: 测试shared_ptr内存引用技术是否已经减1
        //     value_.blob_shared_ptr_value = nullptr;
        // } else if(type_ == ParamType::rawbuffer_shared_ptr_value_enum) {
        //     value_.rawbuffer_shared_ptr_value = nullptr;
        // }
    };
    inline bool IsEmpty() {
        return type_ == ParamType::default_type_enum;
    };
    DECLARE_PARAMWRAPPER_FUNCS(int, int_value);
    DECLARE_PARAMWRAPPER_FUNCS(bool, bool_value);
    DECLARE_PARAMWRAPPER_FUNCS(long, long_value);
    DECLARE_PARAMWRAPPER_FUNCS(float, float_value);
    DECLARE_PARAMWRAPPER_FUNCS(ElementOpType, element_op_value);
    DECLARE_PARAMWRAPPER_POINTER_FUNCS(Blob, blob_pvalue);
    DECLARE_PARAMWRAPPER_POINTER_FUNCS(BaseLayer, base_layer_pvalue);
    DECLARE_PARAMWRAPPER_POINTER_FUNCS(LayerResource, layer_resource_pvalue);
    DECLARE_PARAMWRAPPER_POINTER_FUNCS(AbstractNetwork, network_pvalue);
    DECLARE_PARAMWRAPPER_POINTER_FUNCS(LayerParam, layer_param_pvalue);
    DECLARE_PARAMWRAPPER_POINTER_FUNCS(Blob2BlobMap, blob_2_blob_pvalue);
    DECLARE_PARAMWRAPPER_POINTER_FUNCS(Blob2RawBufferMap, blob_raw_buffer_pvalue);
    DECLARE_PARAMWRAPPER_POINTER_FUNCS(RawBuffer, raw_buffer_pvalue);
    DECLARE_PARAMWRAPPER_POINTER_FUNCS(void, void_pvalue);
    ParamWrapper(bfp16_t bfp16_value) {
        bfp16_value_ = bfp16_value;
        type_        = ParamType::bfp16_value_enum;
    };

    inline bfp16_t GetBfp16() {
        assert(type_ == ParamType::bfp16_value_enum);
        return bfp16_value_;
    };

    inline bool IsBfp16() {
        return type_ == ParamType::bfp16_value_enum;
    };

    ParamWrapper(std::shared_ptr<RawBuffer> raw_buffer_shared_ptr) {
        raw_buffer_shared_ptr_ = raw_buffer_shared_ptr;
        type_                  = ParamType::rawbuffer_shared_ptr_value_enum;
    };
    inline std::shared_ptr<RawBuffer> GetRawbufferSharedPtr() {
        assert(type_ == ParamType::rawbuffer_shared_ptr_value_enum);
        return raw_buffer_shared_ptr_;
    };
    inline bool IsRawbufferSharedPtr() {
        return type_ == ParamType::rawbuffer_shared_ptr_value_enum;
    };

    ParamWrapper(std::shared_ptr<Blob> blob_shared_ptr) {
        blob_shared_ptr_ = blob_shared_ptr;
        type_            = ParamType::blob_shared_ptr_value_enum;
    };
    inline std::shared_ptr<Blob> GetBlobSharedPtr() {
        assert(type_ == ParamType::blob_shared_ptr_value_enum);
        return blob_shared_ptr_;
    };
    inline bool IsBlobSharedPtr() {
        return type_ == ParamType::blob_shared_ptr_value_enum;
    };

    int GetBlobOrRawbufferSize() {
        assert(IsBlobOrRawbuffer());
        BlobDesc desc;
        switch (type_) {
        case blob_shared_ptr_value_enum:
            desc = blob_shared_ptr_->GetBlobDesc();
            return (desc.dims.size() > 0 ? CalculateElementCount(desc) : 0) *
                   DataTypeUtils::GetBytesSize(desc.data_type);
        case blob_pvalue_enum:
            desc = value_.blob_pvalue->GetBlobDesc();
            return (desc.dims.size() > 0 ? CalculateElementCount(desc) : 0) *
                   DataTypeUtils::GetBytesSize(desc.data_type);
        case rawbuffer_shared_ptr_value_enum:
            return raw_buffer_shared_ptr_->GetBytesSize();
        case raw_buffer_pvalue_enum:
            return value_.raw_buffer_pvalue->GetBytesSize();
        default:
            break;
        }
        return 0;
    };

    DimsVector GetBlobOrRawbufferDims() {
        assert(IsBlobOrRawbuffer());
        switch (type_) {
        case blob_shared_ptr_value_enum:
            return blob_shared_ptr_->GetBlobDesc().dims;
        case blob_pvalue_enum:
            return value_.blob_pvalue->GetBlobDesc().dims;
        case rawbuffer_shared_ptr_value_enum:
            return raw_buffer_shared_ptr_->GetBufferDims();
        case raw_buffer_pvalue_enum:
            return value_.raw_buffer_pvalue->GetBufferDims();
        default:
            break;
        }
        return {};
    };

    DataType GetBlobOrRawbufferDatatype() {
        assert(IsBlobOrRawbuffer());
        switch (type_) {
        case blob_shared_ptr_value_enum:
            return blob_shared_ptr_->GetBlobDesc().data_type;
        case blob_pvalue_enum:
            return value_.blob_pvalue->GetBlobDesc().data_type;
        case rawbuffer_shared_ptr_value_enum:
            return raw_buffer_shared_ptr_->GetDataType();
        case raw_buffer_pvalue_enum:
            return value_.raw_buffer_pvalue->GetDataType();
        default:
            break;
        }
        return DataType::DATA_TYPE_AUTO;
    };

    DataFormat GetBlobOrRawbufferDataformat() {
        assert(IsBlobOrRawbuffer());
        switch (type_) {
        case blob_shared_ptr_value_enum:
            return blob_shared_ptr_->GetBlobDesc().data_format;
        case blob_pvalue_enum:
            return value_.blob_pvalue->GetBlobDesc().data_format;
        case rawbuffer_shared_ptr_value_enum:
            return raw_buffer_shared_ptr_->GetDataFormat();
        case raw_buffer_pvalue_enum:
            return value_.raw_buffer_pvalue->GetDataFormat();
        default:
            break;
        }
        return DataFormat::DATA_FORMAT_AUTO;
    };

    void *GetBlobOrRawbufferDataPointer() {
        assert(IsBlobOrRawbuffer());
        switch (type_) {
        case blob_shared_ptr_value_enum:
            return static_cast<void *>(static_cast<char *>(blob_shared_ptr_->GetHandle().base) +
                                       blob_shared_ptr_->GetHandle().bytes_offset);
        case blob_pvalue_enum:
            return static_cast<void *>(static_cast<char *>(value_.blob_pvalue->GetHandle().base) +
                                       value_.blob_pvalue->GetHandle().bytes_offset);
        case rawbuffer_shared_ptr_value_enum:
            return raw_buffer_shared_ptr_->force_to<void *>();
        case raw_buffer_pvalue_enum:
            return value_.raw_buffer_pvalue->force_to<void *>();
        default:
            break;
        }
        return nullptr;
    };

    inline bool IsBlobOrRawbuffer() {
        return type_ == ParamType::blob_shared_ptr_value_enum || type_ == ParamType::blob_pvalue_enum ||
               type_ == ParamType::rawbuffer_shared_ptr_value_enum || type_ == ParamType::raw_buffer_pvalue_enum;
    };

private:
    ParamValue value_;
    ParamType type_;
    std::shared_ptr<Blob> blob_shared_ptr_; // shared_ptr 有默认的构造函数  最好不要放在union里会有未知的行为
    std::shared_ptr<RawBuffer> raw_buffer_shared_ptr_;
    bfp16_t bfp16_value_;
};
typedef std::vector<ParamWrapper> ParamWrappers;

} // namespace train
} // namespace TNN_NS

#endif // TNN_SOURCE_TNN_TRAIN_GRAD_UTILS_H