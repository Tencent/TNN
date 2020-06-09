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

#include "network_helpers.h"

#include "tnn/core/abstract_device.h"
#include "tnn/core/common.h"
#include "tnn/core/context.h"
#include "tnn/layer/base_layer.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

Status BlobHandleAllocate(Blob* blob, AbstractDevice* device) {
    Status ret                   = TNN_OK;
    BlobDesc desc                = blob->GetBlobDesc();
    BlobMemorySizeInfo size_info = device->Calculate(desc);
    void* data;
    ret = device->Allocate(&data, size_info);
    if (ret != TNN_OK) {
        return ret;
    }
    BlobHandle handle;
    handle.base = data;
    blob->SetHandle(handle);
    return ret;
}

Status BlobHandleFree(Blob* blob, AbstractDevice* device) {
    return device->Free(blob->GetHandle().base);
}

DataFormat GetDefaultDataFormat(DeviceType device_type) {
    if (device_type == DEVICE_OPENCL) {
        return DATA_FORMAT_NHC4W4;
    } else if (device_type == DEVICE_METAL || device_type == DEVICE_ARM) {
        return DATA_FORMAT_NC4HW4;
    } else {
        return DATA_FORMAT_NCHW;
    }
}

}  // namespace TNN_NS
