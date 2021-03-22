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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(GroupNorm, LAYER_GROUP_NORM);

template<int THREAD_PER_BLOCK, typename T>
__global__ void group_norm_kernel(const T* input, T* output, const float * gamma,
        const float * beta, const int size, const int batch_size, const int channels_per_group,
        const int group, const int channels, const float eps) {
    __shared__ double ssum1[THREAD_PER_BLOCK/32];
    __shared__ double ssum2[THREAD_PER_BLOCK/32];
    __shared__ double k;
    __shared__ double b;
    extern __shared__ float sm[];

    const int block_offset = (blockIdx.x * channels + blockIdx.y * channels_per_group) * size;
    const T * ptr = input + block_offset;
    T * dst = output + block_offset;

    double thread_sum1 = 0.f;
    double thread_sum2 = 0.f;

    for (int i = threadIdx.x; i < channels_per_group * size; i+=THREAD_PER_BLOCK) {
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
        double mean = thread_sum1 / (size * channels_per_group) ;
        double var = thread_sum2 / (size * channels_per_group) - mean * mean;

        k = 1.f / sqrt(var + eps);
        b = - mean * k;;
    }

    __syncthreads();
    for (int c = threadIdx.x; c < channels_per_group; c+=THREAD_PER_BLOCK) {
        float scale = gamma[blockIdx.y * channels_per_group + c];
        float bias = beta == nullptr ? 0.f : beta[blockIdx.y * channels_per_group + c];
        sm[c] = k * scale;
        sm[channels_per_group+c] = bias + b * scale;
    }
    __syncthreads();
    for (int c = 0; c < channels_per_group; c++) {
        float scale = sm[c];
        float bias = sm[channels_per_group + c];
        for (int i = threadIdx.x; i < size; i += THREAD_PER_BLOCK) {
             dst[c*size+i] = convert_float_value<T>((get_float_value<T>(ptr[c*size+i]) * scale + bias));
        }
    }
}

Status CudaGroupNormLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaGroupNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaGroupNormLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<GroupNormLayerParam*>(param_);
    Blob *input_blob = inputs[0];
    Blob *scale_blob = inputs[1];
    Blob *bias_blob  = inputs[2];
    Blob *output_blob = outputs[0];
    auto input_dims = inputs[0]->GetBlobDesc().dims;
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* scale_data = static_cast<float*>(scale_blob->GetHandle().base);
    float* bias_data  = static_cast<float*>(bias_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    int channels_per_group = input_dims[1] / params->group;

    dim3 grid(input_dims[0], params->group);
    const int THREAD_PER_BLOCK = 128;
    int sm_size = channels_per_group * 2 * sizeof(float);
    group_norm_kernel<THREAD_PER_BLOCK, float><<<grid, THREAD_PER_BLOCK, sm_size, context_->GetStream()>>>(input_data,
        output_data, scale_data, bias_data, input_dims[2]*input_dims[3], input_dims[0], channels_per_group, params->group,
        input_dims[1], params->eps);

    return TNN_OK;
}

REGISTER_CUDA_ACC(GroupNorm, LAYER_GROUP_NORM);

}  // namespace TNN_NS

