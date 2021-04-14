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

#include "tnn/device/arm/arm_device.h"

#include <stdlib.h>

#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

static inline void *armMalloc(size_t size) {
#if _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void *ptr = 0;
    if (posix_memalign(&ptr, 32, size))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(32, size);
#else
    return malloc(size);
#endif
}

ArmDevice::ArmDevice(DeviceType device_type) : AbstractDevice(device_type) {}

ArmDevice::~ArmDevice() {}

BlobMemorySizeInfo ArmDevice::Calculate1DMemorySize(BlobDesc &desc) {
    BlobMemorySizeInfo info;
    info.data_type = desc.data_type;
    int count      = 0;
    if (desc.data_type == DATA_TYPE_HALF) {
        count = desc.dims[0] * ROUND_UP(desc.dims[1], 8) * desc.dims[2] * desc.dims[3];
    } else {
        count = desc.dims[0] * ROUND_UP(desc.dims[1], 4) * desc.dims[2] * desc.dims[3];
    }
    info.dims.push_back(count);
    return info;
}

BlobMemorySizeInfo ArmDevice::Calculate(BlobDesc &desc) {
    return this->Calculate1DMemorySize(desc);
}

Status ArmDevice::Allocate(void **handle, MatType mat_type, DimsVector dims) {
    BlobDesc desc;
    desc.dims        = dims;
    desc.device_type = DEVICE_ARM;
    desc.data_format = DATA_FORMAT_NCHW;
    if (mat_type == NCHW_FLOAT) {
        desc.data_type = DATA_TYPE_FLOAT;
    } else {
        desc.data_type = DATA_TYPE_INT8;
    }
    auto size_info = Calculate(desc);
    return Allocate(handle, size_info);
}

Status ArmDevice::Allocate(void **handle, BlobMemorySizeInfo &size_info) {
    if (handle) {
        int bytes_size = GetBlobMemoryBytesSize(size_info);
        *handle        = armMalloc(bytes_size + NEON_KERNEL_EXTRA_LOAD);
    }
    return TNN_OK;
}

Status ArmDevice::Free(void *handle) {
    if (handle) {
        free(handle);
    }
    return TNN_OK;
}

Status ArmDevice::CopyToDevice(BlobHandle *dst, const BlobHandle *src, BlobDesc &desc, void *command_queue) {
    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    memcpy(GetBlobHandlePtr(*dst), GetBlobHandlePtr(*src), size_in_bytes);

    return TNN_OK;
}

Status ArmDevice::CopyFromDevice(BlobHandle *dst, const BlobHandle *src, BlobDesc &desc, void *command_queue) {
    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    memcpy(GetBlobHandlePtr(*dst), GetBlobHandlePtr(*src), size_in_bytes);

    return TNN_OK;
}

AbstractLayerAcc *ArmDevice::CreateLayerAcc(LayerType type) {
    auto &layer_creator_map = GetLayerCreatorMap();
    if (layer_creator_map.count(type) > 0) {
        return layer_creator_map[type]->CreateLayerAcc(type);
    }
    return NULL;
}

std::shared_ptr<const ImplementedPrecision> ArmDevice::GetImplementedPrecision(LayerType type) {
    auto &layer_precision_map = GetLayerPrecisionMap();
    if (layer_precision_map.count(type) > 0) {
        return layer_precision_map[type];
    }
    return std::make_shared<ImplementedPrecision>();
}

Context *ArmDevice::CreateContext(int device_id) {
    return new ArmContext();
}

Status ArmDevice::RegisterLayerAccCreator(LayerType type, LayerAccCreator *creator) {
    GetLayerCreatorMap()[type] = std::shared_ptr<LayerAccCreator>(creator);
    return TNN_OK;
}

std::map<LayerType, std::shared_ptr<LayerAccCreator>> &ArmDevice::GetLayerCreatorMap() {
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>> layer_creator_map;
    return layer_creator_map;
}

Status ArmDevice::RegisterLayerPrecision(LayerType type, std::shared_ptr<ImplementedPrecision> precision) {
    GetLayerPrecisionMap()[type] = precision;
    return TNN_OK;
}

std::map<LayerType, std::shared_ptr<ImplementedPrecision>> &ArmDevice::GetLayerPrecisionMap() {
    static std::map<LayerType, std::shared_ptr<ImplementedPrecision>> layer_precision_map;
    return layer_precision_map;
};

TypeDeviceRegister<ArmDevice> g_arm_device_register(DEVICE_ARM);

}  // namespace TNN_NS
