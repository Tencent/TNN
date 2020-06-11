// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/cuda_context.h"
#include "device/cuda/cuda_macro.h"

namespace TNN_NS {

CudaContext::~CudaContext() {
    Status ret = this->DeInit();
}

Status CudaContext::LoadLibrary(std::vector<std::string> path) {
    return TNN_OK;
}

Status CudaContext::GetCommandQueue(void** command_queue) {
    *command_queue = stream_;
    return TNN_OK;
}

Status CudaContext::Init(int device_id) {
    device_id_ = device_id;

    CUDA_CHECK(cudaSetDevice(device_id_));

    CUDA_CHECK(cudaStreamCreate(&stream_));

    cudnnStatus_t cudnn_status = cudnnCreate(&cudnn_handle_);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOGE("create cudnn handle failed");
        return TNNERR_INST_ERR;
    }

    cudnn_status = cudnnSetStream(cudnn_handle_, stream_);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOGE("cudnn handle set stream failed");
        return TNNERR_INST_ERR;
    }

    cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        LOGE("create cublas handle failed");
        return TNNERR_INST_ERR;
    }

    cublas_status = cublasSetStream(cublas_handle_, stream_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        LOGE("cublas handle set stream failed");
        return TNNERR_INST_ERR;
    }

    return TNN_OK;
}

Status CudaContext::DeInit() {
    cudaError_t cuda_status = cudaStreamDestroy(stream_);
    if (cuda_status != cudaSuccess) {
        LOGE("destroy cuda stream failed");
    }

    cudnnStatus_t cudnn_status = cudnnDestroy(cudnn_handle_);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        LOGE("destroy cudnn handle failed");
    }

    cublasStatus_t cublas_status = cublasDestroy(cublas_handle_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        LOGE("destroy cublas handle failed");
    }

    return TNN_OK;
}

Status CudaContext::OnInstanceForwardBegin() {
    return TNN_OK;
}

Status CudaContext::OnInstanceForwardEnd() {
    return TNN_OK;
}

Status CudaContext::Synchronize() {
    cudaError_t cuda_status = cudaStreamSynchronize(stream_);
    if (cuda_status != cudaSuccess) {
        LOGE("cuda strema synchronize failed\n");
        return TNNERR_INST_ERR;
    }

    return TNN_OK;
}

}  // namespace TNN_NS
