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
#include "tnn/device/cuda/cuda_macro.h"

namespace TNN_NS {

CudaContext::~CudaContext() {
    cudaError_t status = cudaStreamDestroy(stream_);
    if (cudaSuccess != status) {
        LOGE("destroy cuda stream failed");
    }
}

Status CudaContext::Setup(int device_id) {
    this->device_id_ = device_id;

    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&stream_));

    return TNN_OK;
}

Status CudaContext::LoadLibrary(std::vector<std::string> path) {
    return TNN_OK;
}
 
Status CudaContext::GetCommandQueue(void** command_queue) {
    *command_queue = stream_;
    return TNN_OK;
}

Status CudaContext::OnInstanceForwardBegin() {
    return TNN_OK;
}
Status CudaContext::OnInstanceForwardEnd() {
    return TNN_OK;
}

Status CudaContext::Synchronize() {
    cudaError_t status = cudaStreamSynchronize(stream_);
    if (cudaSuccess != status) {
        LOGE("cuda strema synchronize failed\n");
        return TNNERR_INST_ERR;
    }
    return TNN_OK;
}

}  //  namespace TNN_NS
