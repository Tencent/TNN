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

DECLARE_CUDA_ACC(LayerNorm, LAYER_LAYER_NORM);

template<int THREAD_PER_BLOCK, typename T, typename Acc>
__global__ void layer_norm_kernel(const T * input, T* output, const T *scale,
        const T *bias, const int size, const int batch_size, const float eps) {
    __shared__ Acc ssum1[THREAD_PER_BLOCK/32];
    __shared__ Acc ssum2[THREAD_PER_BLOCK/32];
    __shared__ Acc mean;
    __shared__ Acc var;

    const int block_offset = blockIdx.x * size;
    const T *ptr = input + block_offset;
    T *dst = output + block_offset;

    Acc thread_sum1 = 0.f;
    Acc thread_sum2 = 0.f;

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
        mean = thread_sum1 / size;
        var = (thread_sum2 / size - mean * mean);
        var = 1.0 / sqrt(var + eps);
    }
    __syncthreads();

    #pragma unroll(4)
    for (int i = threadIdx.x; i < size; i += THREAD_PER_BLOCK) {
        float k = get_float_value<T>(scale[i]) * var;
        float b = - mean * k + get_float_value<T>(bias[i]);
        dst[i] = convert_float_value<T>((get_float_value<T>(ptr[i]) * k + b));
    }
}

Status CudaLayerNormLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }

    return TNN_OK;
}

Status CudaLayerNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaLayerNormLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *scale_blob  = inputs[1];
    Blob *bias_blob   = inputs[2];
    Blob *output_blob = outputs[0];

    auto layer_param = dynamic_cast<LayerNormLayerParam *>(param_);
    auto dims_input = input_blob->GetBlobDesc().dims;
    const int reduce_dim_size = layer_param->reduce_dims_size;

    if (layer_param->reduce_dims_size != scale_blob->GetBlobDesc().dims.size()) {
        return Status(TNNERR_PARAM_ERR, "LayerNormLayer has invalid dims for input blob of scale or bias");
    }

    const int channel_dim_size = (int)dims_input.size() - reduce_dim_size;

    const int channels = DimsVectorUtils::Count(dims_input, 0, channel_dim_size);
    const int channel_area = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, channel_dim_size);
    if (0 == channels || 0 == channel_area) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    void *input_data  = input_blob->GetHandle().base;
    void *output_data = output_blob->GetHandle().base;
    void *scale_data  = scale_blob->GetHandle().base;
    void *bias_data   = bias_blob->GetHandle().base;

    const int THREAD_PER_BLOCK = 128;
    dim3 griddim;
    griddim.x = channels;

    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        layer_norm_kernel<THREAD_PER_BLOCK, float, float><<<griddim, THREAD_PER_BLOCK, 0, context_->GetStream()>>>((float*)input_data,
            (float *)output_data, (float *)scale_data, (float *)bias_data, channel_area, channels, layer_param->eps);
    } else if (input_blob->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        layer_norm_kernel<THREAD_PER_BLOCK, __half, float><<<griddim, THREAD_PER_BLOCK, 0, context_->GetStream()>>>((__half*)input_data,
            (__half *)output_data, (__half *)scale_data, (__half *)bias_data, channel_area, channels, layer_param->eps);
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc don't support datatype");
    }
    return TNN_OK;
}

REGISTER_CUDA_ACC(LayerNorm, LAYER_LAYER_NORM);

}  // namespace TNN_NS
