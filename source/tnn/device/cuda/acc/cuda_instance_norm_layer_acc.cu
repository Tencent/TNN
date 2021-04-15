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

#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_radix_sort.cuh>

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(InstanceNorm, LAYER_INST_BATCH_NORM);

template<int THREAD_PER_BLOCK, typename T>
__global__ void instance_norm_kernel(const T * input, T* output, const float * gamma,
        const float * beta, const int size, const int batch_size, const int C, const float eps) {
    __shared__ double ssum1[THREAD_PER_BLOCK/32];
    __shared__ double ssum2[THREAD_PER_BLOCK/32];
    __shared__ float k;
    __shared__ float b;

    // const int batch_offset = blockIdx.y * size;
    const int block_offset = blockIdx.x * size;
    const T * ptr = input + block_offset;
    T * dst = output + block_offset;
    const int cid = blockIdx.x % C;
    
    double thread_sum1 = 0.f;
    double thread_sum2 = 0.f;

    for (int i = threadIdx.x; i < size; i+=THREAD_PER_BLOCK) {
        float value = get_float_value<T>(ptr[i]);
        thread_sum1 += value;
        thread_sum2 += value * value;
    }

    thread_sum1 += __shfl_down_sync(0xffffffff, thread_sum1, 16, 32);
    thread_sum1 += __shfl_down_sync(0x0000ffff, thread_sum1, 8, 16);
    thread_sum1 += __shfl_down_sync(0x000000ff, thread_sum1, 4, 8);
    thread_sum1 += __shfl_down_sync(0x0000000f, thread_sum1, 2, 4);
    thread_sum1 += __shfl_down_sync(0x00000003, thread_sum1, 1, 2);

    thread_sum2 += __shfl_down_sync(0xffffffff, thread_sum2, 16, 32);
    thread_sum2 += __shfl_down_sync(0x0000ffff, thread_sum2, 8, 16);
    thread_sum2 += __shfl_down_sync(0x000000ff, thread_sum2, 4, 8);
    thread_sum2 += __shfl_down_sync(0x0000000f, thread_sum2, 2, 4);
    thread_sum2 += __shfl_down_sync(0x00000003, thread_sum2, 1, 2);

    if (threadIdx.x % 32 == 0) {
        ssum1[threadIdx.x / 32] = thread_sum1;
        ssum2[threadIdx.x / 32] = thread_sum2;
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x / 32) {
        thread_sum1 = ssum1[threadIdx.x];
        thread_sum2 = ssum2[threadIdx.x];
    } else {
        thread_sum1 = 0;
        thread_sum2 = 0;
    }
    thread_sum1 += __shfl_down_sync(0x0000000f, thread_sum1, 2, 4);
    thread_sum1 += __shfl_down_sync(0x00000003, thread_sum1, 1, 2);

    thread_sum2 += __shfl_down_sync(0x0000000f, thread_sum2, 2, 4);
    thread_sum2 += __shfl_down_sync(0x00000003, thread_sum2, 1, 2);

    if (threadIdx.x == 0) {
        double mean = thread_sum1 / size;
        double var = thread_sum2 / size - mean * mean;

        k = gamma[cid] / sqrt(var + eps);
        b = - mean * k + beta[cid];
    }
    
    __syncthreads();
    #pragma unroll(4)
    for (int i = threadIdx.x; i < size; i += THREAD_PER_BLOCK) {
        dst[i] = convert_float_value<T>((get_float_value<T>(ptr[i]) * k + b));
    }
}

Status CudaInstanceNormLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }

    auto res = dynamic_cast<InstanceNormLayerResource *>(resource);
    if (!res) {
        LOGE("Error: InstanceNormLayerResource is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: InstanceNormLayerResource is nil");
    }

    float *k_data = res->scale_handle.force_to<float *>();
    int k_size = res->scale_handle.GetBytesSize();
    float *b_data = res->bias_handle.force_to<float *>();
    int b_size = res->bias_handle.GetBytesSize();

    CreateTempBuf(k_size);
    CreateTempBuf(b_size);
    cudaMemcpyAsync(tempbufs_[0].ptr, k_data, k_size, cudaMemcpyHostToDevice, context_->GetStream());
    cudaMemcpyAsync(tempbufs_[1].ptr, b_data, b_size, cudaMemcpyHostToDevice, context_->GetStream());
    return TNN_OK;
}

Status CudaInstanceNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaInstanceNormLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto dims = input_blob->GetBlobDesc().dims;
    int num = dims[0];
    int channels = dims[1];
    int height = dims[2];
    int width = dims[3];
    int count = DimsVectorUtils::Count(dims);
    int hw = DimsVectorUtils::Count(dims, 2);
    void* input_data = input_blob->GetHandle().base;
    void* output_data = output_blob->GetHandle().base;

    const int THREAD_PER_BLOCK = 128;
    dim3 griddim;
    griddim.x = channels * num;

    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        instance_norm_kernel<THREAD_PER_BLOCK, float><<<griddim, THREAD_PER_BLOCK, 0, context_->GetStream()>>>((float*)input_data,
            (float*)output_data, (const float *)tempbufs_[0].ptr, (const float *)tempbufs_[1].ptr, hw, channels * num, channels, 1e-5);
    } else if (input_blob->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        instance_norm_kernel<THREAD_PER_BLOCK, __half><<<griddim, THREAD_PER_BLOCK, 0, context_->GetStream()>>>((__half*)input_data,
            (__half*)output_data, (const float *)tempbufs_[0].ptr, (const float *)tempbufs_[1].ptr, hw, channels * num, channels, 1e-5);
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc don't support datatype");
    }
    return TNN_OK;
}

REGISTER_CUDA_ACC(InstanceNorm, LAYER_INST_BATCH_NORM);

}  // namespace TNN_NS
