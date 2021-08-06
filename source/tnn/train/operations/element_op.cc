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

#include "tnn/train/operations/base_op.h"
#include "tnn/train/grad/utils.h"
//special for nchw to nc4hw4 transform
#include "tnn/device/arm/arm_util.h"

namespace TNN_NS {
namespace train {
DECLARE_OP(Element, ElementOp);

template <typename T, ElementOpType type>
void calculate(T *out, T* in, int count) {
    return;
}
template <typename T>
void calculate_div_nc4hw4(T* out, T* in, int batch, int channel, int hw) {
    return;
};
template <>
void calculate<float, ElementOpType::Add>(float *out, float* in, int count) {
    for(size_t i=0; i<count; ++i) {
        out[i] += in[i];
    }
}
template <>
void calculate<float, ElementOpType::Sub>(float *out, float* in, int count) {
    for(size_t i=0; i<count; ++i) 
        out[i] = in[i] - out[i];
}
template <>
void calculate<float, ElementOpType::Mul>(float *out, float* in, int count) {
    for(size_t i=0; i<count; ++i) 
        out[i] *= in[i];
}
template <>
void calculate<float, ElementOpType::Div>(float *out, float* in, int count) {
    //note don't use this in nc4hw4 format
    for(size_t i=0; i<count; ++i) 
        out[i] = in[i] / out[i];
}
template <>
void calculate<float, ElementOpType::Log>(float *out, float* in, int count) {
    for(size_t i=0; i<count; ++i) 
        out[i] = std::log(in[i]);
}
template <>
void calculate<float, ElementOpType::Neg>(float *out, float* in, int count) {
    for(size_t i=0; i<count; ++i) 
        out[i] = -in[i];
}

template <>
void calculate<bfp16_t, ElementOpType::Add>(bfp16_t *out, bfp16_t* in, int count) {
    cvt_32b t1;
    cvt_32b t2;
    for(size_t i=0; i<count; ++i) {
        t1.u = in[i].w << 16;
        t2.u = out[i].w << 16;
        t2.f += t1.f;
        out[i].w = t2.u >> 16;
    }
}
template <>
void calculate<bfp16_t, ElementOpType::Sub>(bfp16_t *out, bfp16_t* in, int count) {
    cvt_32b t1;
    cvt_32b t2;
    for(size_t i=0; i<count; ++i) {
        t1.u = in[i].w << 16;
        t2.u = out[i].w << 16;
        t2.f = t1.f - t2.f;
        out[i].w = t2.u >> 16;
    }
}
template <>
void calculate<bfp16_t, ElementOpType::Mul>(bfp16_t *out, bfp16_t* in, int count) {
    cvt_32b t1;
    cvt_32b t2;
    cvt_32b t3;
    for(size_t i=0; i<count; ++i) {
        t1.u = in[i].w << 16;
        t2.u = out[i].w << 16;
        t2.f *= t1.f;
        out[i].w = t3.u >> 16;
    }
}
template <>
void calculate<bfp16_t, ElementOpType::Div>(bfp16_t *out, bfp16_t* in, int count) {
    cvt_32b t1;
    cvt_32b t2;
    for(size_t i=0; i<count; ++i) {
        t1.u = in[i].w << 16;
        t2.u = out[i].w << 16;
        t2.f = t1.f / t2.f;
        out[i].w = t2.u >> 16;
    }
}
template <>
void calculate<bfp16_t, ElementOpType::Log>(bfp16_t *out, bfp16_t* in, int count) {
    cvt_32b t1;
    for(size_t i=0; i<count; ++i) {
        t1.u = in[i].w << 16;
        t1.f = std::log(t1.f);
        out[i].w = t1.u >> 16;
    }
}
template <>
void calculate<bfp16_t, ElementOpType::Neg>(bfp16_t *out, bfp16_t* in, int count) {
    cvt_32b t1;
    for(size_t i=0; i<count; ++i) {
        t1.u = in[i].w << 16;
        t1.f = -t1.f;
        out[i].w = t1.u >> 16;
    }
}
template<>
void calculate_div_nc4hw4(bfp16_t* out, bfp16_t* in, int batch, int channel, int hw) {
    // skip empty element in nc4hw4 format, other wise zero may be dived
    cvt_32b t1;
    cvt_32b t2;
    int c, cur_hw;
    int cur = 0;
    for (int n = 0; n < batch; ++n) {
        auto out_ptr_n = out + n * ROUND_UP(channel, 4) * hw;
        auto in_ptr_n = out + n * ROUND_UP(channel, 4) * hw;
        cur = 0;
        for (c = 0; c < channel; ++c) {
            int plane      = c / 4;
            auto *outPlane = plane * hw * 4 + out_ptr_n;
            auto *inPlane = plane * hw * 4 + in_ptr_n;
            int offset     = c % 4;
            for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
                cur = 4 * cur_hw + offset;
                t1.u = inPlane[cur].w << 16;
                t2.u = outPlane[cur].w << 16;
                t2.f = t1.f / t2.f;
                outPlane[cur].w = t2.u >> 16;
            }
        }
    }

}

template<>
void calculate_div_nc4hw4(float* out, float* in, int batch, int channel, int hw) {
    // skip empty element in nc4hw4 format, other wise zero may be dived
    for (int n = 0; n < batch; ++n) {
        auto out_ptr_n = out + n * ROUND_UP(channel, 4) * hw;
        auto in_ptr_n = out + n * ROUND_UP(channel, 4) * hw;
        int c, cur_hw;
        int idx = 0;
        int cur = 0;
        for (c = 0; c < channel; ++c) {
            int plane      = c / 4;
            auto *outPlane = plane * hw * 4 + out_ptr_n;
            auto *inPlane = plane * hw * 4 + in_ptr_n;
            int offset     = c % 4;
            for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
                cur = 4 * cur_hw + offset;
                outPlane[cur] = inPlane[cur] / outPlane[cur] ;
            }
        }
    }

}

bool ElementOp::IsSupported(ParamWrappers& inputs, ParamWrappers& outputs, TrainContext& context) {
    DefaultNetwork* network = dynamic_cast<DefaultNetwork*>(context.network);
    if(network == nullptr)
        return false;
    auto device_type = context.config->device_type;
    // Only avalibale on cpu
    if(device_type != DEVICE_ARM && device_type != DEVICE_NAIVE && device_type != DEVICE_X86)
        return false;
    if(inputs.size() < 1 || outputs.size() < 1)
        return false;
    // Don't support broadcast so far
    if(!DimsVectorUtils::Equal(inputs[0].GetBlobOrRawbufferDims(), outputs[0].GetBlobOrRawbufferDims()))
        return false;
    // Don't support broadcast so far
    if(inputs.size() > 1 && !DimsVectorUtils::Equal(inputs[0].GetBlobOrRawbufferDims(), inputs[1].GetBlobOrRawbufferDims()))
        return false;
    DataType data_type = inputs[0].GetBlobOrRawbufferDatatype();
    DataFormat data_format = inputs[0].GetBlobOrRawbufferDataformat();
    for (size_t i = 0; i < inputs.size(); i++)
    {
        if(!inputs[i].IsBlobOrRawbuffer())
            return false;
        data_type = inputs[i].GetBlobOrRawbufferDatatype();
        if( data_type != DATA_TYPE_BFP16 && data_type != DATA_TYPE_FLOAT) 
            return false;
        //don't support diffrence data format for now
        if(data_format != inputs[i].GetBlobOrRawbufferDataformat()) 
            return false;
        data_format = inputs[i].GetBlobOrRawbufferDataformat();
        if( data_type != DATA_FORMAT_NCHW && data_type != DATA_FORMAT_NC4HW4) 
            return false;
    }
    for (size_t i = 0; i < outputs.size(); i++)
    {
        if(!outputs[i].IsBlobOrRawbuffer())
            return false;
        data_type = outputs[i].GetBlobOrRawbufferDatatype();
        if(data_type != DATA_TYPE_BFP16 && data_type != DATA_TYPE_FLOAT) 
            return false;
        //don't support diffrent data format for now
        if(data_format != outputs[i].GetBlobOrRawbufferDataformat()) 
            return false;
        data_format = outputs[i].GetBlobOrRawbufferDataformat();
        if(data_type != DATA_FORMAT_NCHW && data_type != DATA_FORMAT_NC4HW4) 
            return false;
    }
    return true;
}

/**
 * @brief calcute base math op, note to chack param realtype
 * @param inputs is Blob pointer or rawbuffer pointer
 * @param outputs is rawbuffer pointer
 * @param other_params[0] is element op type
 */
Status ElementOp::Exec(ParamWrappers& inputs, ParamWrappers& outputs, ParamWrappers& other_params) {
    if(other_params.size() <= 0 || !other_params[0].IsElementOpType())
        return Status(TNN_OP_ERROR, "ElementOp param error");
    ElementOpType op_type = other_params[0].GetElementOpType();
    void* output_ptr = outputs[0].GetBlobOrRawbufferDataPointer();
    // calcute by element, so ignore diffrence between NCHW and NC4HW4
    if(!output_ptr) {
        return Status(TNN_OP_ERROR, "output data pointer empty");
    }
    //已经在IsSupported检查过类型匹配
    auto dst_dtype = outputs[0].GetBlobOrRawbufferDatatype();
    auto dst_dformat = outputs[0].GetBlobOrRawbufferDataformat();
    std::vector<ParamWrapper> buffers;
    buffers.resize(inputs.size());
    for(size_t i = 0; i < inputs.size(); i++) {
        auto src_dtype = inputs[i].GetBlobOrRawbufferDatatype();
        auto src_dformat = outputs[0].GetBlobOrRawbufferDataformat();
        if(src_dtype != dst_dtype) {
            auto data_count = inputs[i].GetBlobOrRawbufferSize()/ DataTypeUtils::GetBytesSize(inputs[i].GetBlobOrRawbufferDatatype());
            if(src_dtype == DATA_TYPE_FLOAT && dst_dtype == DATA_TYPE_BFP16) {
                std::shared_ptr<RawBuffer> buffer = std::make_shared<RawBuffer>(data_count * sizeof(bfp16_t), inputs[i].GetBlobOrRawbufferDims());
                buffer->SetDataType(DATA_TYPE_BFP16);
                buffer->SetDataFormat(inputs[i].GetBlobOrRawbufferDataformat());
                ConvertFromFloatToBFP16(static_cast<float*>(inputs[i].GetBlobOrRawbufferDataPointer()), buffer->force_to<void*>(), data_count);
                buffers.emplace_back(ParamWrapper(buffer));
            } else if(src_dtype == DATA_TYPE_BFP16 && dst_dtype == DATA_TYPE_FLOAT) {
                std::shared_ptr<RawBuffer> buffer = std::make_shared<RawBuffer>(data_count*sizeof(float), inputs[i].GetBlobOrRawbufferDims());
                buffer->SetDataType(DATA_TYPE_FLOAT);
                buffer->SetDataFormat(inputs[i].GetBlobOrRawbufferDataformat());
                ConvertFromBFP16ToFloat(static_cast<void*>(inputs[i].GetBlobOrRawbufferDataPointer()), buffer->force_to<float*>(), data_count);
                buffers.emplace_back(ParamWrapper(buffer));
            } else {
                return Status(TNN_OP_ERROR, "data type transformer error");
            }
        } else {
            buffers.push_back(inputs[i]);
        }
        // nchw -> nc4hw4 
        // int PackFloatBlob(float *dst, float *src, size_t batch, size_t channel, size_t hw);
        // nc4hw4 -> nchw
        // int UnpackFloatBlob(float *dst, float *src, size_t batch, size_t channel, size_t hw);
        // int PackFloatBlob(bfp16_t *dst, bfp16_t *src, size_t batch, size_t channel, size_t hw);
        // int UnpackFloatBlob(bfp16_t *dst, bfp16_t *src, size_t batch, size_t channel, size_t hw);

        if(src_dformat != dst_dformat) {
            auto dims = buffers[i].GetBlobOrRawbufferDims();
            int batch             = dims[0];
            int channel           = dims[1];
            int hw                = DimsVectorUtils::Count(dims, 2);
            std::shared_ptr<RawBuffer> buffer = std::make_shared<RawBuffer>(buffers[i].GetBlobOrRawbufferSize(), buffers[i].GetBlobOrRawbufferDims());
            buffer->SetDataType(buffers[i].GetBlobOrRawbufferDatatype());
            if(src_dformat == DATA_FORMAT_NCHW && dst_dformat == DATA_FORMAT_NC4HW4) {
                buffer->SetDataFormat(DATA_FORMAT_NC4HW4);
                if(dst_dtype == DATA_TYPE_BFP16)
                    PackFloatBlob(static_cast<bfp16_t*>(buffers[i].GetBlobOrRawbufferDataPointer()), buffer->force_to<bfp16_t*>(), batch, channel, hw);
                else if(dst_dtype == DATA_TYPE_FLOAT)
                    PackFloatBlob(static_cast<float*>(buffers[i].GetBlobOrRawbufferDataPointer()), buffer->force_to<float*>(), batch, channel, hw);
                else
                    return Status(TNN_OP_ERROR, "data format transfer error, unkown data type");
            } else if(src_dformat == DATA_FORMAT_NC4HW4 && dst_dformat == DATA_FORMAT_NCHW) {
                std::shared_ptr<RawBuffer> buffer = std::make_shared<RawBuffer>(buffers[i].GetBlobOrRawbufferSize(), buffers[i].GetBlobOrRawbufferDims());
                buffer->SetDataType(buffers[i].GetBlobOrRawbufferDatatype());
                buffer->SetDataFormat(DATA_FORMAT_NC4HW4);
                if(dst_dtype == DATA_TYPE_BFP16)
                    UnpackFloatBlob(static_cast<bfp16_t*>(buffers[i].GetBlobOrRawbufferDataPointer()), buffer->force_to<bfp16_t*>(), batch, channel, hw);
                else if(dst_dtype == DATA_TYPE_FLOAT)
                    UnpackFloatBlob(static_cast<float*>(buffers[i].GetBlobOrRawbufferDataPointer()), buffer->force_to<float*>(), batch, channel, hw);
                else
                    return Status(TNN_OP_ERROR, "data format transfer error, unkown data type2");                
            } else {
                return Status(TNN_OP_ERROR, "data format transfer error, unsupport data format");
            }
            buffers[i] = buffer;
        } else {
            // do nothing
        }
    }
    
    int data_count = outputs[0].GetBlobOrRawbufferSize()/ DataTypeUtils::GetBytesSize(outputs[0].GetBlobOrRawbufferDatatype());
    void* input_ptr = buffers[0].GetBlobOrRawbufferDataPointer();
    auto dims = outputs[0].GetBlobOrRawbufferDims();
    if(buffers.size() > 2) {
        memcpy(output_ptr, buffers[1].GetBlobOrRawbufferDataPointer(), outputs[0].GetBlobOrRawbufferSize());
    }
    if(dst_dformat == DATA_FORMAT_NHC4W4 && op_type == ElementOpType::Div) {
        //special deal
        switch (dst_dtype)
        {
            case DATA_TYPE_FLOAT:
                calculate_div_nc4hw4<float>(static_cast<float*>(output_ptr), static_cast<float*>(input_ptr), dims[0], dims[1], DimsVectorUtils::Count(dims, 2));
                break;
            case DATA_TYPE_BFP16:
                calculate_div_nc4hw4<float>(static_cast<float*>(output_ptr), static_cast<float*>(input_ptr), dims[0], dims[1], DimsVectorUtils::Count(dims, 2));
                break;
            default:
                return Status(TNN_OP_ERROR, "Unsupport data type ");
        }
        return Status(TNN_OK);
    } 
    switch (dst_dtype)
    {
        case DATA_TYPE_FLOAT:
            switch (op_type)
            {
                case ElementOpType::Add:
                    calculate<float, ElementOpType::Add>(static_cast<float*>(input_ptr), static_cast<float*>(output_ptr), data_count);
                    break;
                case ElementOpType::Sub:
                    calculate<float, ElementOpType::Sub>(static_cast<float*>(input_ptr), static_cast<float*>(output_ptr), data_count);
                    break;
                case ElementOpType::Mul:
                    calculate<float, ElementOpType::Mul>(static_cast<float*>(input_ptr), static_cast<float*>(output_ptr), data_count);
                    break;
                case ElementOpType::Div:
                    calculate<float, ElementOpType::Div>(static_cast<float*>(input_ptr), static_cast<float*>(output_ptr), data_count);
                    break;
                case ElementOpType::Log:
                    calculate<float, ElementOpType::Log>(static_cast<float*>(input_ptr), static_cast<float*>(output_ptr), data_count);
                    break;
                case ElementOpType::Neg:
                    calculate<float, ElementOpType::Neg>(static_cast<float*>(input_ptr), static_cast<float*>(output_ptr), data_count);
                    break;
                default:
                    return Status(TNN_OP_ERROR, "Unsupport ElementOpType ");
            }
            break;
        case DATA_TYPE_BFP16:
            switch (op_type)
            {
                case ElementOpType::Add:
                    calculate<bfp16_t, ElementOpType::Add>(static_cast<bfp16_t*>(input_ptr), static_cast<bfp16_t*>(output_ptr), data_count);
                    break;
                case ElementOpType::Sub:
                    calculate<bfp16_t, ElementOpType::Sub>(static_cast<bfp16_t*>(input_ptr), static_cast<bfp16_t*>(output_ptr), data_count);
                    break;
                case ElementOpType::Mul:
                    calculate<bfp16_t, ElementOpType::Mul>(static_cast<bfp16_t*>(input_ptr), static_cast<bfp16_t*>(output_ptr), data_count);
                    break;
                case ElementOpType::Div:
                    calculate<bfp16_t, ElementOpType::Div>(static_cast<bfp16_t*>(input_ptr), static_cast<bfp16_t*>(output_ptr), data_count);
                    break;
                case ElementOpType::Log:
                    calculate<bfp16_t, ElementOpType::Log>(static_cast<bfp16_t*>(input_ptr), static_cast<bfp16_t*>(output_ptr), data_count);
                    break;
                case ElementOpType::Neg:
                    calculate<bfp16_t, ElementOpType::Neg>(static_cast<bfp16_t*>(input_ptr), static_cast<bfp16_t*>(output_ptr), data_count);
                    break;
                default:
                    return Status(TNN_OP_ERROR, "Unsupport ElementOpType");
            }
            break;
        default:
            return Status(TNN_OP_ERROR, "Unsupport data type ");
    }
    return Status(TNN_OK);

}

} //namespace train         
} //namspace TNN_NS