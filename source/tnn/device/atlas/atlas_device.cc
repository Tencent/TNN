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

#include "tnn/device/atlas/atlas_device.h"
#include "acl/ops/acl_dvpp.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

AtlasDevice::AtlasDevice(DeviceType device_type) : AbstractDevice(device_type) {}

AtlasDevice::~AtlasDevice() {}

BlobMemorySizeInfo AtlasDevice::Calculate(BlobDesc& desc) {
    return Calculate1DMemorySize(desc);
}

Status AtlasDevice::Allocate(void** handle, MatType mat_type, DimsVector dims) {
    if (dims.size() != 4) {
        LOGE("invalid dim size: %d\n", (int)dims.size());
        return Status(TNNERR_PARAM_ERR, "invalid dim size");
    }

    BlobMemorySizeInfo size_info;

    int N = dims[0];
    int C = dims[1];
    int H = dims[2];
    int W = dims[3];

    if (NCHW_FLOAT == mat_type) {
        size_info.data_type = DATA_TYPE_FLOAT;
        size_info.dims.push_back(N * C * W * H);
    } else if (N8UC3 == mat_type) {
        size_info.data_type = DATA_TYPE_INT8;
        size_info.dims.push_back(N * 3 * W * H);
    } else if (N8UC4 == mat_type) {
        size_info.data_type = DATA_TYPE_INT8;
        size_info.dims.push_back(N * 4 * W * H);
    } else if (NGRAY == mat_type) {
        size_info.data_type = DATA_TYPE_INT8;
        size_info.dims.push_back(N * 1 * W * H);
    } else if (NNV12 == mat_type) {
        size_info.data_type = DATA_TYPE_INT8;
        size_info.dims.push_back(N * 3 * W * H / 2);
    } else if (NNV21 == mat_type) {
        size_info.data_type = DATA_TYPE_INT8;
        size_info.dims.push_back(N * 3 * W * H / 2);
    } else {
        LOGE("atlas allocator not support this mat type: %d\n", mat_type);
        return Status(TNNERR_PARAM_ERR, "not support this mat type");
    }

    return Allocate(handle, size_info);
}

// allocate atlas memory
Status AtlasDevice::Allocate(void** handle, BlobMemorySizeInfo& size_info) {
    ASSERT(size_info.dims.size() == 1);

    int bytes_size = GetBlobMemoryBytesSize(size_info);
    if (bytes_size == 0) {
        return Status(TNNERR_PARAM_ERR, "invalid size for memory allocate");
    }

    aclError ret = acldvppMalloc(handle, bytes_size);
    if (ret != ACL_ERROR_NONE) {
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "atlas alloc memory failed");
    }
    LOGD("atlas allocate memory addr: 0x%lx\n", *handle);
    return TNN_OK;
}

// release atlas memory
Status AtlasDevice::Free(void* handle) {
    aclError ret = acldvppFree(handle);
    if (ret != ACL_ERROR_NONE) {
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "atlas free memory failed");
    }
    return TNN_OK;
}

// Copy data from Cpu To Device, format is same.
Status AtlasDevice::CopyToDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue) {
    return Status(TNNERR_DEVICE_NOT_SUPPORT, "Atlas not support CopyToDevice");
}

// Copy data from Device To Cpu, format is same.
Status AtlasDevice::CopyFromDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue) {
    return Status(TNNERR_DEVICE_NOT_SUPPORT, "Atlas not support CopyFromDevice");
}

// create layer acc with layer type
AbstractLayerAcc* AtlasDevice::CreateLayerAcc(LayerType type) {
    return nullptr;
}

Context* AtlasDevice::CreateContext(int) {
    return nullptr;
}

NetworkType AtlasDevice::ConvertAutoNetworkType() {
    return NETWORK_TYPE_ATLAS;
}

TypeDeviceRegister<AtlasDevice> g_atlas_device_register(DEVICE_ATLAS);

}  // namespace TNN_NS
