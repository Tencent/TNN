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

#include "tnn/train/grad/utils.h"
#include "tnn/core/macro.h"
#include "tnn/device/arm/arm_util.h"

namespace TNN_NS {
namespace train {
void ConvertToNCHW(void *&src_ptr, RawBuffer &dst, const BlobDesc &input_desc) {
    ConvertToNCHW(src_ptr, dst, input_desc.data_type, input_desc.data_format, input_desc.dims);
}

void ConvertToNCHW(void *&src_ptr, RawBuffer &dst, RawBuffer *input_rawbuffer) {
    ConvertToNCHW(src_ptr, dst, input_rawbuffer->GetDataType(), input_rawbuffer->GetDataFormat(),
                  input_rawbuffer->GetBufferDims());
}

// src_ptr must be nchw format
// TODO: error return for judging status
void ConvertToNCHW(void *&src_ptr, RawBuffer &dst, const DataType &dtype, const DataFormat &dformat,
                   const DimsVector &dims) {
    if (dformat == DATA_FORMAT_NC4HW4) {
        int total_count      = DimsVectorUtils::Count(dims);
        int batch_for_pack   = dims[0];
        int channel_for_pack = dims[1];
        int hw_for_pack      = DimsVectorUtils::Count(dims, 2);
        dst                  = RawBuffer(total_count * DataTypeUtils::GetBytesSize(dtype), dims);
        if (dtype == DATA_TYPE_BFP16) {
            UnpackFloatBlob(dst.force_to<bfp16_t *>(), static_cast<bfp16_t *>(src_ptr), batch_for_pack,
                            channel_for_pack, hw_for_pack);
            src_ptr = dst.force_to<void *>();
        } else if (dtype == DATA_TYPE_FLOAT) {
            UnpackFloatBlob(dst.force_to<float *>(), static_cast<float *>(src_ptr), batch_for_pack, channel_for_pack,
                            hw_for_pack);
            src_ptr = dst.force_to<void *>();
        }
    }
}

void ConvertToNC4HW4(std::shared_ptr<RawBuffer> &src, BlobDesc &input_desc) {
    if (input_desc.data_format == DATA_FORMAT_NC4HW4) {
        int batch_for_pack                   = input_desc.dims[0];
        int channel_for_pack                 = input_desc.dims[1];
        int hw_for_pack                      = DimsVectorUtils::Count(input_desc.dims, 2);
        std::shared_ptr<RawBuffer> tmpbuffer = std::make_shared<RawBuffer>(
            CalculateElementCount(input_desc) * DataTypeUtils::GetBytesSize(input_desc.data_type), input_desc.dims);
        if (input_desc.data_type == DATA_TYPE_BFP16) {
            PackFloatBlob(tmpbuffer->force_to<bfp16_t *>(), src->force_to<bfp16_t *>(), batch_for_pack,
                          channel_for_pack, hw_for_pack);
        } else if (input_desc.data_type == DATA_TYPE_FLOAT) {
            PackFloatBlob(tmpbuffer->force_to<float *>(), src->force_to<float *>(), batch_for_pack, channel_for_pack,
                          hw_for_pack);
        }
        tmpbuffer->SetDataFormat(input_desc.data_format);
        tmpbuffer->SetDataType(input_desc.data_type);
        src = tmpbuffer;
    }
}

int ConvertFromBFP16ToFloat(void *fp16, float *fp32, int count) {
    bfp16_t *bfp16PTR = (bfp16_t *)fp16;
    for (int i = 0; i < count; ++i) {
        fp32[i] = float(bfp16PTR[i]);
    }

    return 0;
}

int ConvertFromFloatToBFP16(float *fp32, void *fp16, int count) {
    bfp16_t *bfp16PTR = (bfp16_t *)fp16;
    for (int i = 0; i < count; ++i) {
        bfp16PTR[i] = fp32[i];
    }

    return 0;
}

int GetDim(const DimsVector dims, const int index) {
    return dims.size() > index ? dims[index] : 1;
};

int CalculateElementCount(BlobDesc &desc) {
    int count = 1;
    if (desc.data_format == DATA_FORMAT_NCHW || desc.data_format == DATA_FORMAT_AUTO) {
        for (auto d : desc.dims)
            count *= d;
    } else {
        // packed format
        if (desc.data_type == DATA_TYPE_HALF) {
            count = GetDim(desc.dims, 0) * ROUND_UP(GetDim(desc.dims, 1), 8) * DimsVectorUtils::Count(desc.dims, 2);
        } else {
            count = GetDim(desc.dims, 0) * ROUND_UP(GetDim(desc.dims, 1), 4) * DimsVectorUtils::Count(desc.dims, 2);
        }
    }
    return count;
}

} // namespace train
} // namespace TNN_NS