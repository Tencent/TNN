// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/cuda_device.h"

#include <cuda_runtime.h>

#include "device/cuda/cuda_context.h"
#include "device/cuda/cuda_macro.h"
#include "utils/data_type_utils.h"
#include "utils/blob_memory_size_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

CudaDevice::CudaDevice(DeviceType device_type) : AbstractDevice(device_type) {}

CudaDevice::~CudaDevice() {}

BlobMemorySizeInfo CudaDevice::Calculate(BlobDesc& desc) {
    return Calculate1DMemorySize(desc);
}

Status CudaDevice::Allocate(void** handle, BlobMemorySizeInfo& size_info) {
    void* ptr;
    int bytes_size     = GetBlobMemoryBytesSize(size_info);
    cudaError_t status = cudaMalloc(&ptr, bytes_size);
    if (status != cudaSuccess) {
        return TNNERR_OUTOFMEMORY;
    }

    LOGD("cuda alloc size:%lu %X\n", bytes_size, ptr);

    *handle = ptr;
    return TNN_OK;
}

Status CudaDevice::Free(void* handle) {
    cudaError_t status = cudaFree(handle);
    if (status != cudaSuccess) {
        LOGE("cuda free error");
        return TNNERR_COMMON_ERROR;
    }
    return TNN_OK;
}

Status CudaDevice::CopyToDevice(BlobHandle* dst, const BlobHandle* src,
                                BlobDesc& desc, void* command_queue) {
    cudaStream_t stream = static_cast<cudaStream_t>(command_queue);
    if (stream == nullptr) {
        return Status(TNNERR_DEVICE_INVALID_COMMAND_QUEUE);
    }

    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    cudaError_t status =
        cudaMemcpyAsync(reinterpret_cast<char*>(dst->base) + dst->bytes_offset,
                        reinterpret_cast<char*>(src->base) + src->bytes_offset,
                        size_in_bytes, cudaMemcpyHostToDevice, stream);

    CUDA_CHECK(status);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return TNN_OK;
}

Status CudaDevice::CopyFromDevice(BlobHandle* dst, const BlobHandle* src,
                                  BlobDesc& desc, void* command_queue) {
    cudaStream_t stream = static_cast<cudaStream_t>(command_queue);
    if (stream == nullptr) {
        return TNNERR_DEVICE_INVALID_COMMAND_QUEUE;
    }

    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);

    cudaError_t status =
        cudaMemcpyAsync(reinterpret_cast<char*>(dst->base) + dst->bytes_offset,
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
    Status ret   = context->Init(device_id);
    if (ret != TNN_OK) {
        LOGE("Cuda context init failed\n");
        delete context;
        return NULL;
    }
    return context;
}

Status CudaDevice::RegisterLayerAccCreator(
    LayerType type, std::shared_ptr<LayerAccCreator> creator) {
    GetLayerCreatorMap()[type] = creator;
    return TNN_OK;
}

std::map<LayerType, std::shared_ptr<LayerAccCreator>>&
CudaDevice::GetLayerCreatorMap() {
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>>
        layer_creator_map;
    return layer_creator_map;
}

TypeDeviceRegister<CudaDevice> g_cuda_device_register(DEVICE_CUDA);

}  // namespace TNN_NS
