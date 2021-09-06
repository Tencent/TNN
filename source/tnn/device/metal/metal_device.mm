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

#import <Metal/Metal.h>

#include "tnn/core/blob.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/device/metal/metal_device.h"
#include "tnn/device/metal/metal_macro.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/device/metal/acc/metal_cpu_adapter_acc.h"

namespace TNN_NS {

BlobMemorySizeInfo MetalDevice::Calculate1DMemorySize(BlobDesc &desc) {
    BlobMemorySizeInfo info;
    info.data_type = desc.data_type;
    int count      = 0;
    if (desc.data_format == DATA_FORMAT_NC4HW4) {
        count = desc.dims[0] * ROUND_UP(DimsFunctionUtils::GetDim(desc.dims, 1), 4) * DimsVectorUtils::Count(desc.dims, 2);
    } else {
        count = DimsVectorUtils::Count(desc.dims);
    }
    info.dims.push_back(count);
    return info;
}

MetalDevice::MetalDevice(DeviceType device_type) : AbstractDevice(device_type) {}

MetalDevice::~MetalDevice() {}

BlobMemorySizeInfo MetalDevice::Calculate(BlobDesc &desc) {
    return MetalDevice::Calculate1DMemorySize(desc);
}

Status MetalDevice::Allocate(void **handle, MatType mat_type, DimsVector dims) {
    if (!handle) {
        return TNN_OK;
    }

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    if (mat_type == N8UC4) {
        int dims_size               = (int)dims.size();
        int height                  = dims[dims_size - 2];
        int width                   = dims[dims_size - 1];
        auto textureDescriptor      = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                    width:width
                                                                                   height:height
                                                                                mipmapped:NO];
        textureDescriptor.usage     = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        id<MTLTexture> texture_rgba = [device newTextureWithDescriptor:textureDescriptor];
        *handle                     = (void *)CFBridgingRetain(texture_rgba);
        return TNN_OK;
    } else if (mat_type == NCHW_FLOAT) {
        BlobDesc desc;
        desc.data_type   = DATA_TYPE_FLOAT;
        desc.dims        = dims;
        desc.device_type = DEVICE_METAL;
        desc.data_format = DATA_FORMAT_NCHW;
        auto size_info   = Calculate(desc);
        int size         = GetBlobMemoryBytesSize(size_info);
        auto buffer      = [device newBufferWithLength:size options:MTLResourceCPUCacheModeDefaultCache];
        *handle          = (void *)CFBridgingRetain(buffer);
        return TNN_OK;
    } else if (mat_type == NC_INT32) {
        BlobDesc desc;
        desc.data_type   = DATA_TYPE_INT32;
        desc.dims        = dims;
        desc.device_type = DEVICE_METAL;
        desc.data_format = DATA_FORMAT_NCHW;
        auto size_info   = Calculate(desc);
        int size         = GetBlobMemoryBytesSize(size_info);
        auto buffer      = [device newBufferWithLength:size options:MTLResourceCPUCacheModeDefaultCache];
        *handle          = (void *)CFBridgingRetain(buffer);
        return TNN_OK;
    } else {
        LOGE("unsupport mat type: %d", mat_type);
        return Status(TNNERR_PARAM_ERR, "unsupport mat type");
    }
}

Status MetalDevice::Allocate(void **handle, BlobMemorySizeInfo &size_info) {
    if (handle) {
        int size             = GetBlobMemoryBytesSize(size_info);
        id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
#if TNN_METAL_DEBUG
        id<MTLBuffer> buffer = [device newBufferWithLength:size options:MTLResourceCPUCacheModeDefaultCache];
#else
        id<MTLBuffer> buffer = [device newBufferWithLength:size options:MTLResourceStorageModePrivate];
#endif
        *handle = (void *)CFBridgingRetain(buffer);
    }
    return TNN_OK;
}

Status MetalDevice::Free(void *handle) {
    if (handle) {
        CFBridgingRelease(handle);
    }
    return TNN_OK;
}

Status MetalDevice::CopyToDevice(BlobHandle *dst, const BlobHandle *src, BlobDesc &desc, void *command_queue) {

    if (!command_queue) {
        LOGD("command_queue is nil context\n");
        return Status(TNNERR_CONTEXT_ERR, "Error: command_queue is nil context");
    }

    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(dst->base);
    uint64_t offset      = dst->bytes_offset;
    memcpy((char *)buffer.contents + offset, ((char *)src->base) + src->bytes_offset, size_in_bytes);
    //    LOGE("inputdata gpu: %.6f %.6f\n", ((float *)buffer.contents)[0], ((float *)buffer.contents)[1]);
    return TNN_OK;
}

Status MetalDevice::CopyFromDevice(BlobHandle *dst, const BlobHandle *src, BlobDesc &desc, void *command_queue) {
    if (!command_queue) {
        LOGD("command_queue is nil context\n");
        return Status(TNNERR_CONTEXT_ERR, "Error: command_queue is nil context");
    }

    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(src->base);
    uint64_t offset      = src->bytes_offset;
    memcpy(((char *)dst->base) + dst->bytes_offset, (char *)buffer.contents + offset, size_in_bytes);

    //    LOGE("outputdata gpu: %.6f %.6f\n", ((float *)buffer.contents)[0], ((float *)buffer.contents)[1]);
    return TNN_OK;
}

AbstractLayerAcc *MetalDevice::CreateLayerAcc(LayerType type) {
    std::map<LayerType, std::shared_ptr<LayerAccCreator>> &layer_creator_map = GetLayerCreatorMap();
    if (layer_creator_map.count(type) > 0) {
        return layer_creator_map[type]->CreateLayerAcc(type);
    } else {
        LOGD("There is no metal layer acc with type %d, now we will try MetalCpuAdapterAcc to forword it on cpu\n", type);
        return new MetalCpuAdapterAcc(type);
    }
    return NULL;
}

NetworkType MetalDevice::ConvertAutoNetworkType() {
    return NETWORK_TYPE_DEFAULT;
}

Context *MetalDevice::CreateContext(int device_id) {
    return new MetalContext();
}

std::shared_ptr<const ImplementedLayout> MetalDevice::GetImplementedLayout(LayerType type) {
    auto &layer_layout_map = GetLayerLayoutMap();
    if (layer_layout_map.count(type) > 0) {
        return layer_layout_map[type];
    }
    return std::make_shared<ImplementedLayout>();
}

std::map<LayerType, std::shared_ptr<LayerAccCreator>> &MetalDevice::GetLayerCreatorMap() {
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>> layer_creator_map;
    return layer_creator_map;
}

std::map<LayerType, std::shared_ptr<ImplementedLayout>> &MetalDevice::GetLayerLayoutMap() {
    static std::map<LayerType, std::shared_ptr<ImplementedLayout>> layer_layout_map;
    return layer_layout_map;
}

Status MetalDevice::RegisterLayerAccCreator(LayerType type, LayerAccCreator *creator) {
    std::map<LayerType, std::shared_ptr<LayerAccCreator>> &layer_creator_map = GetLayerCreatorMap();
    layer_creator_map[type]                                   = std::shared_ptr<LayerAccCreator>(creator);
    return TNN_OK;
}

Status MetalDevice::RegisterLayerLayout(LayerType type, std::shared_ptr<ImplementedLayout> layout) {
    GetLayerLayoutMap()[type] = layout;
    return TNN_OK;
}

TypeDeviceRegister<MetalDevice> g_metal_device_register(DEVICE_METAL);

} // namespace TNN_NS
