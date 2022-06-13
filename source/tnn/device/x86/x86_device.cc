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

#include "tnn/device/x86/acc/x86_cpu_adapter_acc.h"
#include "tnn/device/x86/x86_device.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

X86Device::X86Device(DeviceType device_type) : AbstractDevice(device_type) {}

X86Device::~X86Device() {}

BlobMemorySizeInfo X86Device::Calculate1DMemorySize(BlobDesc &desc) {
    BlobMemorySizeInfo info;
    info.data_type = desc.data_type;
    int count      = 0;
    if (desc.data_type == DATA_TYPE_INT8) {
        count = desc.dims[0] * ROUND_UP(desc.dims[1], 4) * DimsVectorUtils::Count(desc.dims, 2);
    } else {
        count = DimsVectorUtils::Count(desc.dims);
    }
    info.dims.push_back(count);
    return info;
}

BlobMemorySizeInfo X86Device::Calculate(BlobDesc &desc) {
    return this->Calculate1DMemorySize(desc);
}

Status X86Device::Allocate(void** handle, MatType mat_type, DimsVector dims) {
    BlobDesc desc;
    desc.dims = dims;
    desc.device_type = DEVICE_X86;
    if (mat_type == NCHW_FLOAT || mat_type == RESERVED_BFP16_TEST || mat_type == RESERVED_INT8_TEST) {
        desc.data_type   = DATA_TYPE_FLOAT;
        desc.data_format = DATA_FORMAT_NCHW;
        auto size_info   = Calculate(desc);
        return Allocate(handle, size_info);
    } else if (mat_type == N8UC3 || mat_type == N8UC4 || mat_type == NGRAY ||
               mat_type == NNV21 || mat_type == NNV12) {
        desc.data_type   = DATA_TYPE_INT8;
        desc.data_format = DATA_FORMAT_NCHW;
        auto size_info   = Calculate(desc);
        return Allocate(handle, size_info);
    } else {
        LOGE("X86Device dont support mat_type:%d", mat_type);
        return Status(TNNERR_PARAM_ERR, "x86 dont support mat_type");
    }
}

Status X86Device::Allocate(void** handle, BlobMemorySizeInfo& size_info) {
    if (handle) {
        *handle = malloc(GetBlobMemoryBytesSize(size_info));
    }
    return TNN_OK;
}

Status X86Device::Free(void* handle) {
    if (handle) {
        free(handle);
    }
    return TNN_OK;
}

std::shared_ptr<const ImplementedLayout> X86Device::GetImplementedLayout(LayerType type) {
    auto layouts = new ImplementedLayout();
    layouts->layouts.push_back(DATA_FORMAT_NCHW);
    return std::shared_ptr<ImplementedLayout>(layouts);
}

Status X86Device::CopyToDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue) {
    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    memcpy(reinterpret_cast<char*>(dst->base) + dst->bytes_offset,
           reinterpret_cast<char*>(src->base) + src->bytes_offset, size_in_bytes);
    return TNN_OK;
}

Status X86Device::CopyFromDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue) {
    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    memcpy(reinterpret_cast<char*>(dst->base) + dst->bytes_offset,
           reinterpret_cast<char*>(src->base) + src->bytes_offset, size_in_bytes);

    return TNN_OK;
}

AbstractLayerAcc* X86Device::CreateLayerAcc(LayerType type) {
    auto &layer_creator_map = GetLayerCreatorMap();
    if (layer_creator_map.count(type) > 0) {
        return layer_creator_map[type]->CreateLayerAcc(type);
    } else {
        return new X86CpuAdapterAcc(type);
    }
}

Context* X86Device::CreateContext(int device_id) {
    return new X86Context();
}

NetworkType X86Device::ConvertAutoNetworkType() {
    return NETWORK_TYPE_DEFAULT;
}

Status X86Device::RegisterLayerAccCreator(LayerType type, LayerAccCreator* creator) {
    GetLayerCreatorMap()[type] = std::shared_ptr<LayerAccCreator>(creator);
    return TNN_OK;
}

std::map<LayerType, std::shared_ptr<LayerAccCreator>>& X86Device::GetLayerCreatorMap() {
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>> layer_creator_map;
    return layer_creator_map;
}

TypeDeviceRegister<X86Device> g_x86_device_register(DEVICE_X86);

} // namespace TNN_NS
