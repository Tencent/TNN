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

#include <cuda_runtime.h>

#include "tnn/device/cuda/cuda_context.h"
#include "tnn/device/cuda/cuda_device.h"
#include "tnn/device/cuda/cuda_macro.h"
#include "tnn/utils/blob_memory_size_utils.h"

namespace TNN_NS {

CudaDevice::CudaDevice(DeviceType device_type) : AbstractDevice(device_type) {}

CudaDevice::~CudaDevice() {}

BlobMemorySizeInfo CudaDevice::Calculate(BlobDesc& desc) {
    return Calculate1DMemorySize(desc);
}

Status CudaDevice::Allocate(void **handle, MatType mat_type, DimsVector dims) {
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

Status CudaDevice::Allocate(void** handle, BlobMemorySizeInfo& size_info) {
    void* ptr;
    int bytes_size = GetBlobMemoryBytesSize(size_info);
    cudaError_t status = cudaMalloc(&ptr, bytes_size);
    if (cudaSuccess != status) {
        LOGE("cuda alloc failed with size %d for %p status:%d\n", bytes_size, ptr, status);
        return TNNERR_OUTOFMEMORY;
    }

    *handle = ptr;
    return TNN_OK;
}

Status CudaDevice::Allocate(void** handle, size_t size) {
    void* ptr = nullptr;
    cudaError_t status = cudaMalloc(&ptr, size);
    if (cudaSuccess != status) {
        LOGE("cuda alloc failed with size %lu for %p, status:%d\n", size, ptr, status);
        return TNNERR_OUTOFMEMORY;
    }
    if (ptr == nullptr) {
        LOGE("cuda alloc got nullptr\n");
        return TNNERR_OUTOFMEMORY;
    }
    *handle = ptr;
    return TNN_OK;
}

Status CudaDevice::ReAllocate(void** handle, size_t size) {
    Status ret;
    if (*handle != nullptr) {
        ret = Free(*handle);
        if (ret != TNN_OK) {
            return ret;
        }
    }
    ret = Allocate(handle, size);
    return ret;
}

Status CudaDevice::Free(void* handle) {
    cudaError_t status = cudaFree(handle);
    if (cudaSuccess != status) {
        LOGE("cuda free failed.");
        return TNNERR_COMMON_ERROR;
    }
    return TNN_OK;
}

Status CudaDevice::CopyToDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue) {
    cudaStream_t stream = static_cast<cudaStream_t>(command_queue);
    if (nullptr == stream) {
        return Status(TNNERR_DEVICE_INVALID_COMMAND_QUEUE);
    }

    auto size_info = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    cudaError_t status = cudaMemcpyAsync(reinterpret_cast<char*>(dst->base) + dst->bytes_offset,
                        reinterpret_cast<char*>(src->base) + src->bytes_offset,
                        size_in_bytes, cudaMemcpyHostToDevice, stream);

    CUDA_CHECK(status);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return TNN_OK;
}

Status CudaDevice::CopyFromDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue) {
    cudaStream_t stream = static_cast<cudaStream_t>(command_queue);
    if (nullptr == stream) {
        return TNNERR_DEVICE_INVALID_COMMAND_QUEUE;
    }
    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    cudaError_t status = cudaMemcpyAsync(reinterpret_cast<char*>(dst->base) + dst->bytes_offset,
                        reinterpret_cast<char*>(src->base) + src->bytes_offset,
                        size_in_bytes, cudaMemcpyDeviceToHost, stream);

    CUDA_CHECK(status);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return TNN_OK;
}

AbstractLayerAcc* CudaDevice::CreateLayerAcc(LayerType type) {
    auto layer_creator_map = GetLayerCreatorMap();
    if (layer_creator_map.count(type) > 0) {
        AbstractLayerAcc* ptr = layer_creator_map[type]->CreateLayerAcc(type);
        return ptr;
    }
    return NULL;
}

Context* CudaDevice::CreateContext(int device_id) {
    auto context = new CudaContext();
    Status ret = context->Setup(device_id);
    if (TNN_OK != ret) {
        LOGE("Cuda context setup failed.");
        delete context;
        return NULL;
    }
    return context;
}

Status CudaDevice::RegisterLayerAccCreator(LayerType type, LayerAccCreator *creator) {
    GetLayerCreatorMap()[type] = std::shared_ptr<LayerAccCreator>(creator);
    return TNN_OK;
}

std::map<LayerType, std::shared_ptr<LayerAccCreator>>&
CudaDevice::GetLayerCreatorMap() {
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>> layer_creator_map;
    return layer_creator_map;
}

TypeDeviceRegister<CudaDevice> g_cuda_device_register(DEVICE_CUDA);

}  //  namespace TNN_NS
