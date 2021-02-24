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

#include "cuda_fp16.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(CbamFusedPooling, LAYER_CBAM_FUSED_POOLING);

template<int THREAD_PER_BLOCK, typename T>
__global__ void cbam_fused_pooling_kernel(const float *input, T* output, T* output2, int n) {
    __shared__ float smax[THREAD_PER_BLOCK/32];
    __shared__ float ssum[THREAD_PER_BLOCK/32];
    int block_offset = blockIdx.x * n;
    const float4* ptr = (const float4*)(input + block_offset);
    float thread_max = -FLT_MAX;
    float thread_sum = 0;
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
        float4 data = ptr[i];
        thread_max = fmaxf(thread_max, data.x);
        thread_max = fmaxf(thread_max, data.y);
        thread_max = fmaxf(thread_max, data.z);
        thread_max = fmaxf(thread_max, data.w);
        thread_sum += data.x;
        thread_sum += data.y;
        thread_sum += data.z;
        thread_sum += data.w;
    }

    thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, 16, 32));
    thread_max = fmaxf(thread_max, __shfl_down_sync(0x0000ffff, thread_max, 8, 16));
    thread_max = fmaxf(thread_max, __shfl_down_sync(0x000000ff, thread_max, 4, 8));
    thread_max = fmaxf(thread_max, __shfl_down_sync(0x0000000f, thread_max, 2, 4));
    thread_max = fmaxf(thread_max, __shfl_down_sync(0x00000003, thread_max, 1, 2));

    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, 16, 32);
    thread_sum += __shfl_down_sync(0x0000ffff, thread_sum, 8, 16);
    thread_sum += __shfl_down_sync(0x000000ff, thread_sum, 4, 8);
    thread_sum += __shfl_down_sync(0x0000000f, thread_sum, 2, 4);
    thread_sum += __shfl_down_sync(0x00000003, thread_sum, 1, 2);

    if (threadIdx.x % 32 == 0) {
        smax[threadIdx.x / 32] = thread_max;
        ssum[threadIdx.x / 32] = thread_sum;
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x / 32) {
        thread_max = smax[threadIdx.x];
        thread_sum = ssum[threadIdx.x];
    } else {
        thread_max = 0;
        thread_sum = 0;
    }
    thread_max = fmaxf(thread_max, __shfl_down_sync(0x0000000f, thread_max, 2, 4));
    thread_max = fmaxf(thread_max, __shfl_down_sync(0x00000003, thread_max, 1, 2));
    thread_sum += __shfl_down_sync(0x0000000f, thread_sum, 2, 4);
    thread_sum += __shfl_down_sync(0x00000003, thread_sum, 1, 2);
    if (threadIdx.x == 0) {
        output[blockIdx.x] = convert_float_value<T>(thread_sum / n);
        output2[blockIdx.x] = convert_float_value<T>(thread_max);
    }
}

template<int THREAD_PER_BLOCK>
__global__ void cbam_fused_pooling_half_kernel(const __half *input, __half* output, __half* output2, int n) {
    __shared__ float smax[THREAD_PER_BLOCK/32];
    __shared__ __half2 ssum[THREAD_PER_BLOCK/32];
    int block_offset = blockIdx.x * n;
    float thread_max = -FLT_MAX;

    __half2 sum = __halves2half2(0, 0);
    if (n % 8 == 0) {
        const float4* ptr = (const float4*)(input + block_offset);
        for (int i = threadIdx.x; i < n / 8; i += blockDim.x) {
            __half2 data[4];
            *((float4*)(&data)) = ptr[i];
            sum = __hadd2(sum, data[0]);
            sum = __hadd2(sum, data[1]);
            sum = __hadd2(sum, data[2]);
            sum = __hadd2(sum, data[3]);
            thread_max = fmaxf(thread_max, __high2float(data[0]));
            thread_max = fmaxf(thread_max, __low2float(data[0]));
            thread_max = fmaxf(thread_max, __high2float(data[1]));
            thread_max = fmaxf(thread_max, __low2float(data[1]));
            thread_max = fmaxf(thread_max, __high2float(data[2]));
            thread_max = fmaxf(thread_max, __low2float(data[2]));
            thread_max = fmaxf(thread_max, __high2float(data[3]));
            thread_max = fmaxf(thread_max, __low2float(data[3]));
        }
    } else {
        const float2* ptr = (const float2*)(input + block_offset);
        for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
            __half2 data[2];
            *((float2*)(&data)) = ptr[i];
            sum = __hadd2(sum, data[0]);
            sum = __hadd2(sum, data[1]);
            thread_max = fmaxf(thread_max, __high2float(data[0]));
            thread_max = fmaxf(thread_max, __low2float(data[0]));
            thread_max = fmaxf(thread_max, __high2float(data[1]));
            thread_max = fmaxf(thread_max, __low2float(data[1]));
        }
    }

    sum = __hadd2(sum, __shfl_down_sync(0xffffffff, sum, 16, 32));
    thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, 16, 32));
    sum = __hadd2(sum, __shfl_down_sync(0x0000ffff, sum, 8, 16));
    thread_max = fmaxf(thread_max, __shfl_down_sync(0x0000ffff, thread_max, 8, 16));
    sum = __hadd2(sum, __shfl_down_sync(0x000000ff, sum, 4, 8));
    thread_max = fmaxf(thread_max, __shfl_down_sync(0x000000ff, thread_max, 4, 8));
    sum = __hadd2(sum, __shfl_down_sync(0x0000000f, sum, 2, 4));
    thread_max = fmaxf(thread_max, __shfl_down_sync(0x0000000f, thread_max, 2, 4));
    sum = __hadd2(sum, __shfl_down_sync(0x00000003, sum, 1, 2));
    thread_max = fmaxf(thread_max, __shfl_down_sync(0x00000003, thread_max, 1, 2));

    if (threadIdx.x % 32 == 0) {
        smax[threadIdx.x / 32] = thread_max;
        ssum[threadIdx.x / 32] = sum;
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x / 32) {
        thread_max = smax[threadIdx.x];
        sum = ssum[threadIdx.x];
    } else {
        thread_max = 0;
        sum = {0, 0};
    }

    thread_max = fmaxf(thread_max, __shfl_down_sync(0x0000000f, thread_max, 2, 4));
    sum = __hadd2(sum, __shfl_down_sync(0x0000000f, sum, 2, 4));
    thread_max = fmaxf(thread_max, __shfl_down_sync(0x00000003, thread_max, 1, 2));
    sum = __hadd2(sum, __shfl_down_sync(0x00000003, sum, 1, 2));

    if (threadIdx.x == 0) {
        output[blockIdx.x] = __float2half((__high2float(sum) + __low2float(sum)) / n);
        output2[blockIdx.x] = thread_max;
    }
}

Status CudaCbamFusedPoolingLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaCbamFusedPoolingLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaCbamFusedPoolingLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *output_blob1 = outputs[0];
    Blob *output_blob2 = outputs[1];

    DataType type = input_blob->GetBlobDesc().data_type;
    DataFormat format = input_blob->GetBlobDesc().data_format;

    int batch_size = input_blob->GetBlobDesc().dims[0];
    int nchannels = input_blob->GetBlobDesc().dims[1];
    int inp_H = input_blob->GetBlobDesc().dims[2];
    int inp_W = input_blob->GetBlobDesc().dims[3];

    const int thread_num = 128;
    int block_num = batch_size * nchannels;
    void* input_ptr = input_blob->GetHandle().base;
    void* output_ptr1 = output_blob1->GetHandle().base;
    void* output_ptr2 = output_blob2->GetHandle().base;

    if (type == DataType::DATA_TYPE_HALF) {
        cbam_fused_pooling_half_kernel<thread_num><<<block_num, thread_num, 0, context_->GetStream()>>>(
            static_cast<__half *>(input_ptr), static_cast<__half *>(output_ptr1),
            static_cast<__half *>(output_ptr2), inp_H * inp_W);
    } else if (type == DataType::DATA_TYPE_FLOAT) {
        if (output_blob1->GetBlobDesc().data_type == DATA_TYPE_FLOAT)
            cbam_fused_pooling_kernel<thread_num, float><<<block_num, thread_num, 0, context_->GetStream()>>>(
                static_cast<float *>(input_ptr), static_cast<float *>(output_ptr1),
                static_cast<float *>(output_ptr2), inp_H * inp_W);
        else
            cbam_fused_pooling_kernel<thread_num, __half><<<block_num, thread_num, 0, context_->GetStream()>>>(
                static_cast<float *>(input_ptr), static_cast<__half *>(output_ptr1),
                static_cast<__half *>(output_ptr2), inp_H * inp_W);
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", type);
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(CbamFusedPooling, LAYER_CBAM_FUSED_POOLING);

}   // namespace TNN_NS
