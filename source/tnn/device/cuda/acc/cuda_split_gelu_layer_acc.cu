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

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_utils.h"

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace TNN_NS {

DECLARE_CUDA_ACC(SplitGelu, LAYER_FUSED_SPLIT_GELU);

using half = __half;

template <typename T, int32_t tHHS, int32_t tTPB>
__global__ void splitGeLUKernel(T const* input, T* output, float const fDivRecip, float const fAdd, float const fMul)
{
    assert(input != nullptr);
    assert(output != nullptr);

    int32_t indexInput = blockIdx.x * tHHS * 2 + threadIdx.x;
    int32_t indexOutput = blockIdx.x * tHHS + threadIdx.x;

#pragma unroll
    for (int32_t i = 0; i < tHHS / tTPB; ++i)
    {
        auto valueL = static_cast<float>(input[indexInput]);
        auto valueR = static_cast<float>(input[indexInput + tHHS]);
        float tmp = valueR;
        tmp *= fDivRecip;
        tmp = erff(tmp);
        tmp += fAdd;
        tmp *= valueR;
        tmp *= fMul;
        tmp *= valueL;
        output[indexOutput] = static_cast<T>(tmp);
        indexInput += tTPB;
        indexOutput += tTPB;
    }
    return;
}

template <typename T>
int32_t launchSplitGeLUKernel(cudaStream_t stream, int32_t gridSize, int32_t nHalfHiddenSize, T const* input, T* output,
    float const fDiv, float const fAdd, float const fMul)
{
    auto const fDivRecip = 1.F / fDiv;
    constexpr int32_t kTPB = 256; // thread per block
    switch (nHalfHiddenSize)
    {
    case 1280: (splitGeLUKernel<T, 1280, kTPB>) <<<gridSize, kTPB, 0, stream>>>(input, output, fDivRecip, fAdd, fMul); break;
    case 2560: (splitGeLUKernel<T, 2560, kTPB>) <<<gridSize, kTPB, 0, stream>>>(input, output, fDivRecip, fAdd, fMul); break;
    case 5120: (splitGeLUKernel<T, 5120, kTPB>) <<<gridSize, kTPB, 0, stream>>>(input, output, fDivRecip, fAdd, fMul); break;
    }
    return 0;
}

template __global__ void splitGeLUKernel<float, 1280, 256>(float const*, float*, float const, float const, float const);
template __global__ void splitGeLUKernel<float, 2560, 256>(float const*, float*, float const, float const, float const);
template __global__ void splitGeLUKernel<float, 5120, 256>(float const*, float*, float const, float const, float const);
template __global__ void splitGeLUKernel<half, 1280, 256>(half const*, half*, float const, float const, float const);
template __global__ void splitGeLUKernel<half, 2560, 256>(half const*, half*, float const, float const, float const);
template __global__ void splitGeLUKernel<half, 5120, 256>(half const*, half*, float const, float const, float const);

template int32_t launchSplitGeLUKernel<float>(cudaStream_t stream, int32_t gridSize, int32_t nHalfHiddenSize,
    float const* input, float* output, float const fDiv, float const fAdd, float const fMul);

template int32_t launchSplitGeLUKernel<half>(cudaStream_t stream, int32_t gridSize, int32_t nHalfHiddenSize,
    half const* input, half* output, float const fDiv, float const fAdd, float const fMul);

Status CudaSplitGeluLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaSplitGeluLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaSplitGeluLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto output_dims = output_blob->GetBlobDesc().dims;
    void *input_data  = input_blob->GetHandle().base;
    void *output_data = output_blob->GetHandle().base;
    auto dtype = input_blob->GetBlobDesc().data_type;

    float div = 1.4140625F;
    float add = 1.F;
    float mul = 0.5F;

    int32_t const grid_size = output_dims[0] * output_dims[1];
    int32_t const half_hidden_size = output_dims[2];

    if (dtype == DATA_TYPE_FLOAT)
    {
        auto const input = static_cast<float const*>(input_data);
        auto output = static_cast<float*>(output_data);
        launchSplitGeLUKernel<float>(context_->GetStream(), grid_size, half_hidden_size, input, output, div, add, mul);
    }
    else
    {
        auto const input = static_cast<half const*>(input_data);
        auto output = static_cast<half*>(output_data);
        launchSplitGeLUKernel<half>(context_->GetStream(), grid_size, half_hidden_size, input, output, div, add, mul);
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(SplitGelu, LAYER_FUSED_SPLIT_GELU);

}  // namespace TNN_NS
