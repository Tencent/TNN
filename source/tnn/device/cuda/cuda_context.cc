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

#include "tnn/device/cuda/cuda_context.h"

#include <cublas_v2.h>
#include <cudnn.h>

#include "tnn/device/cuda/cuda_macro.h"

namespace TNN_NS {

CudaContext::~CudaContext() {
    if (own_stream_) {
        cudaError_t status = cudaStreamDestroy(stream_);
        if (cudaSuccess != status) {
            LOGE("destroy cuda stream failed");
        }
    }

    if(own_cudnn_handle_) {
        cudnnStatus_t cudnn_status = cudnnDestroy(cudnn_handle_);
        if (cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOGE("destroy cudnn handle failed");
        }
    }

    if(own_cublas_handle_) {
        cublasStatus_t cublas_status = cublasDestroy(cublas_handle_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            LOGE("destroy cublas handle failed");
        }
    }

    if (workspace_) {
        CUDA_CHECK(cudaFree(workspace_));
    }
}

Status CudaContext::Setup(int device_id) {
    this->device_id_ = device_id;

    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    own_stream_ = true;
    return TNN_OK;
}

Status CudaContext::LoadLibrary(std::vector<std::string> path) {
    return TNN_OK;
}

Status CudaContext::GetCommandQueue(void** command_queue) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    *command_queue = stream_;
    return TNN_OK;
}

Status CudaContext::SetCommandQueue(void* command_queue) {
    if (own_stream_) {
        CUDA_CHECK(cudaStreamSynchronize(stream_))
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }
    own_stream_ = false;
    stream_ = (cudaStream_t)command_queue;

    if(own_cudnn_handle_) {
        cudnnStatus_t cudnn_status = cudnnSetStream(cudnn_handle_, stream_);
        if (cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOGE("cudnn handle set stream failed");
            return TNNERR_INST_ERR;
        }
    }

    if(own_cublas_handle_) {
        cublasStatus_t cublas_status = cublasSetStream(cublas_handle_, stream_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            LOGE("cublas handle set stream failed");
            return TNNERR_INST_ERR;
        }
    }

    return TNN_OK;
}

Status CudaContext::ShareCommandQueue(Context* context) {

    if (context == nullptr)
        return TNNERR_NULL_PARAM;

    CudaContext* cuda_ctx = dynamic_cast<CudaContext*>(context);
    if (cuda_ctx == nullptr)
        return TNNERR_DEVICE_INVALID_COMMAND_QUEUE;

    if (own_stream_) {
        CUDA_CHECK(cudaStreamSynchronize(stream_))
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }
    own_stream_ = false;
    stream_ = cuda_ctx->GetStream();

    if(own_cudnn_handle_) {
        cudnnStatus_t cudnn_status = cudnnSetStream(cudnn_handle_, stream_);
        if (cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOGE("cudnn handle set stream failed");
            return TNNERR_INST_ERR;
        }
    }

    if(own_cublas_handle_) {
        cublasStatus_t cublas_status = cublasSetStream(cublas_handle_, stream_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            LOGE("cublas handle set stream failed");
            return TNNERR_INST_ERR;
        }
    }

    return TNN_OK;
}

Status CudaContext::OnInstanceForwardBegin() {
    return TNN_OK;
}
Status CudaContext::OnInstanceForwardEnd() {
    return TNN_OK;
}

cudaStream_t& CudaContext::GetStream() {
    return stream_;
}

cudnnHandle_t& CudaContext::GetCudnnHandle() {
    if(!own_cudnn_handle_) {
        cudnnStatus_t cudnn_status = cudnnCreate(&cudnn_handle_);
        if (cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOGE("create cudnn handle failed");
            return cudnn_handle_;
        }
	own_cudnn_handle_ = true;
        cudnn_status = cudnnSetStream(cudnn_handle_, stream_);
        if (cudnn_status != CUDNN_STATUS_SUCCESS) {
            LOGE("cudnn handle set stream failed");
            return cudnn_handle_;
        }
    }
    return cudnn_handle_;
}

cublasHandle_t& CudaContext::GetCublasHandle() {
    if(!own_cublas_handle_) {
        cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            LOGE("create cublas handle failed");
            return cublas_handle_;
        }
	own_cublas_handle_ = true;
        cublas_status = cublasSetStream(cublas_handle_, stream_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            LOGE("cublas handle set stream failed");
            return cublas_handle_;
        }
    }
    return cublas_handle_;
}

void* CudaContext::GetWorkspace() {
    return workspace_;
}

void CudaContext::SetWorkspaceSize(int size) {
    if (size > workspace_size_) {
        if (workspace_) {
            CUDA_CHECK(cudaFree(workspace_));
        }
        CUDA_CHECK(cudaMalloc(&workspace_, size));
        workspace_size_ = size;
    }
}

Status CudaContext::Synchronize() {
    cudaError_t status = cudaStreamSynchronize(stream_);
    if (cudaSuccess != status) {
        LOGE("cuda stream synchronize failed\n");
        return TNNERR_CUDA_SYNC_ERROR;
    }
    return TNN_OK;
}

Status CudaContext::AddQuantResource(std::string name, std::shared_ptr<RawBuffer> res) {
    quant_extra_res_[name] = res;
    return TNN_OK;
}

std::shared_ptr<RawBuffer> CudaContext::GetQuantResource(std::string name) {
    if (quant_extra_res_.count(name) == 0) {
        return nullptr;
    }
    return quant_extra_res_[name];
}

}  //  namespace TNN_NS
