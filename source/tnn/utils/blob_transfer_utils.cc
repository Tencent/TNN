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

}  // namespace TNN_NS
