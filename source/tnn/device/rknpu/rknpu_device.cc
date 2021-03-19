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

#include "tnn/device/rknpu/rknpu_device.h"

#include "tnn/device/rknpu/rknpu_context.h"
#include "tnn/utils/blob_memory_size_utils.h"

namespace TNN_NS {

RknpuDevice::RknpuDevice(DeviceType device_type) : AbstractDevice(device_type) {}

RknpuDevice::~RknpuDevice() {}

BlobMemorySizeInfo RknpuDevice::Calculate(BlobDesc& desc) {
    return Calculate1DMemorySize(desc);
}

Status RknpuDevice::Allocate(void** handle, BlobMemorySizeInfo& size_info) {
    if (handle) {
        *handle = malloc(GetBlobMemoryBytesSize(size_info));
    }
    return TNN_OK;
}

Status RknpuDevice::Allocate(void** handle, MatType mat_type, DimsVector dims) {
    BlobDesc desc;
    desc.dims        = dims;
    desc.device_type = DEVICE_NAIVE;
    if (mat_type == NCHW_FLOAT) {
        desc.data_type   = DATA_TYPE_FLOAT;
        desc.data_format = DATA_FORMAT_NCHW;
        auto size_info   = Calculate(desc);
        return Allocate(handle, size_info);
    } else {
        LOGE("CpuDevice dont support mat_type:%d", mat_type);
        return Status(TNNERR_PARAM_ERR, "cpu dont support mat_type");
    }
}

Status RknpuDevice::Free(void* handle) {
    if (handle) {
        free(handle);
    }
    return TNN_OK;
}

Status RknpuDevice::CopyToDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue) {
    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    memcpy(reinterpret_cast<char*>(dst->base) + dst->bytes_offset,
           reinterpret_cast<char*>(src->base) + src->bytes_offset, size_in_bytes);
    return TNN_OK;
}

Status RknpuDevice::CopyFromDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue) {
    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    memcpy(reinterpret_cast<char*>(dst->base) + dst->bytes_offset,
           reinterpret_cast<char*>(src->base) + src->bytes_offset, size_in_bytes);

    return TNN_OK;
}

AbstractLayerAcc* RknpuDevice::CreateLayerAcc(LayerType type) {
    auto& layer_creator_map = GetLayerCreatorMap();
    if (layer_creator_map.count(type) > 0) {
        return layer_creator_map[type]->CreateLayerAcc(type);
    }
    return NULL;
}

Context* RknpuDevice::CreateContext(int device_id) {
    return new RknpuContext();
}

NetworkType RknpuDevice::ConvertAutoNetworkType() {
    return NETWORK_TYPE_RK_NPU;
}

std::map<LayerType, std::shared_ptr<LayerAccCreator>>& RknpuDevice::GetLayerCreatorMap() {
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>> layer_creator_map;
    return layer_creator_map;
}

TypeDeviceRegister<RknpuDevice> g_rknpu_device_register(DEVICE_RK_NPU);

}  // namespace TNN_NS
