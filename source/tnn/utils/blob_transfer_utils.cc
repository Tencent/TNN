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

#include "tnn/utils/blob_transfer_utils.h"

#include "tnn/core/abstract_device.h"
#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status CopyToDevice(Blob* dst, Blob* src, void* command_queue) {
    DeviceType device_type = dst->GetBlobDesc().device_type;

    Status ret = TNN_OK;

    auto device = GetDevice(device_type);
    if (device == NULL) {
        return TNNERR_DEVICE_NOT_SUPPORT;
    }

    BlobHandle src_handle = src->GetHandle();
    BlobHandle dst_handle = dst->GetHandle();
    BlobDesc blob_desc    = src->GetBlobDesc();

    ret = device->CopyToDevice(&dst_handle, &src_handle, blob_desc, command_queue);

    if (ret != TNN_OK) {
        LOGD("Copy blob to device failed\n");
        return ret;
    }

    return TNN_OK;
}

Status CopyFromDevice(Blob* dst, Blob* src, void* command_queue) {
    DeviceType device_type = src->GetBlobDesc().device_type;

    Status ret = TNN_OK;

    auto device = GetDevice(device_type);

    if (device == NULL) {
        return TNNERR_DEVICE_NOT_SUPPORT;
    }

    BlobHandle src_handle = src->GetHandle();
    BlobHandle dst_handle = dst->GetHandle();
    BlobDesc blob_desc    = src->GetBlobDesc();

    ret = device->CopyFromDevice(&dst_handle, &src_handle, blob_desc, command_queue);

    if (ret != TNN_OK) {
        LOGD("Copy blob to device failed\n");
        return ret;
    }

    return TNN_OK;
}

Status Blob2RawBuffer(Blob *blob, std::shared_ptr<RawBuffer> &buffer) {
    if (!blob) {
        return Status(TNNERR_PARAM_ERR, "blob is null");
    }
    if (blob->GetBlobDesc().device_type != DEVICE_NAIVE) {
        LOGE("Blob2RawBuffer dont support device type: %d", blob->GetBlobDesc().device_type);
        return Status(TNNERR_PARAM_ERR, "Blob2RawBuffer dont support device type");
    }
    
    const auto dims = blob->GetBlobDesc().dims;
    
    int count = DimsVectorUtils::Count(dims);
    if (dims.size() == 0 && !blob->GetHandle().base) {
        count = 0;
    }
    const int ele_size = DataTypeUtils::GetBytesSize(blob->GetBlobDesc().data_type);
    
    //处理原来buffer已有分配内存的情况
    if (!buffer || buffer->GetBytesSize() != count*ele_size) {
        buffer = std::make_shared<RawBuffer>(count*ele_size);
    }
    buffer->SetDataType(blob->GetBlobDesc().data_type);
    buffer->SetBufferDims(blob->GetBlobDesc().dims);
    
    if (count > 0) {
        memcpy(buffer->force_to<void *>(), blob->GetHandle().base, count*ele_size);
    }
    
    return TNN_OK;
}

Status RawBuffer2Blob(RawBuffer *buffer, std::shared_ptr<Blob> &blob) {
    if (!buffer) {
        LOGE("RawBuffer2Blob:: buffer is null \n");
        return Status(TNNERR_PARAM_ERR, "RawBuffer2Blob:: buffer is null");
    }
    
    const int count = blob ? DimsVectorUtils::Count(blob->GetBlobDesc().dims) : 0;
    const int ele_size = blob ? DataTypeUtils::GetBytesSize(blob->GetBlobDesc().data_type) : 0;
    
    if (!blob || buffer->GetBytesSize() != count*ele_size) {
        BlobDesc desc;
        {
            desc.device_type = DEVICE_NAIVE;
            desc.data_type = buffer->GetDataType();
            desc.dims = buffer->GetBufferDims();
        }
        if (buffer->GetBytesSize() > 0) {
            blob = std::make_shared<Blob>(desc, true);
        } else {
            blob = std::make_shared<Blob>(desc, false);
        }
    }
    
    if (blob->GetHandle().base && buffer->GetBytesSize() > 0) {
        memcpy(blob->GetHandle().base, buffer->force_to<void *>(), buffer->GetBytesSize());
    }
    
    return TNN_OK;
}

}  // namespace TNN_NS
