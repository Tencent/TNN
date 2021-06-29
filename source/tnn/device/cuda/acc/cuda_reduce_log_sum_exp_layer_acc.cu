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

namespace TNN_NS {

DECLARE_CUDA_ACC(ReduceLogSumExp, LAYER_REDUCE_LOG_SUM_EXP);

template <int blockSize, typename T>
__global__ void reduce_log_sum_exp_kernel(const int num, const int channels,
        const int spatial_dim, const T* input, T* output) {
    int n = blockIdx.x / spatial_dim;
    int s = blockIdx.x % spatial_dim;

    __shared__ float smax[blockSize/32];
    __shared__ float ssum[blockSize/32];

    int tid = threadIdx.x;
    float max_value = -FLT_MAX;
    for (int c = tid; c < channels; c += blockDim.x) {
        float value = get_float_value<T>(input[(n * channels + c) * spatial_dim + s]);
        max_value = fmaxf(value, max_value);
    }

    float tmp = __shfl_down_sync(0xffffffff, max_value, 16, 32);
    max_value = fmaxf(max_value, tmp);
    tmp = __shfl_down_sync(0xffffffff, max_value, 16, 32);
    max_value = fmaxf(max_value, tmp);
    tmp = __shfl_down_sync(0x0000ffff, max_value, 8, 16);
    max_value = fmaxf(max_value, tmp);
    tmp = __shfl_down_sync(0x000000ff, max_value, 4, 8);
    max_value = fmaxf(max_value, tmp);
    tmp = __shfl_down_sync(0x0000000f, max_value, 2, 4);
    max_value = fmaxf(max_value, tmp);
    tmp = __shfl_down_sync(0x00000003, max_value, 1, 2);
    max_value = fmaxf(max_value, tmp);

    if (tid % 32 == 0) {
        smax[tid / 32] = max_value;
    }
    __syncthreads();

    if (tid < blockDim.x / 32) {
        max_value = smax[tid];
    } else {
        max_value = -FLT_MAX;
    }

    tmp = __shfl_down_sync(0x0000000f, max_value, 2, 4);
    max_value = fmaxf(max_value, tmp);
    tmp = __shfl_down_sync(0x00000003, max_value, 1, 2);
    max_value = fmaxf(max_value, tmp);

    if (tid == 0) {
        smax[0] = max_value;
    }
    __syncthreads();

    float thread_sum = 0;
    for (int c = tid; c < channels; c += blockDim.x) {
        float value = get_float_value<T>(input[(n * channels + c) * spatial_dim + s]);
        thread_sum += exp(value - smax[0]);
    }

    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, 16, 32);
    thread_sum += __shfl_down_sync(0x0000ffff, thread_sum, 8, 16);
    thread_sum += __shfl_down_sync(0x000000ff, thread_sum, 4, 8);
    thread_sum += __shfl_down_sync(0x0000000f, thread_sum, 2, 4);
    thread_sum += __shfl_down_sync(0x00000003, thread_sum, 1, 2);

    if (tid % 32 == 0) {
        ssum[tid / 32] = thread_sum;
    }
    __syncthreads();

    if (tid < blockDim.x / 32) {
        thread_sum = ssum[tid];
    } else {
        thread_sum = 0;
    }

    thread_sum += __shfl_down_sync(0x0000000f, thread_sum, 2, 4);
    thread_sum += __shfl_down_sync(0x00000003, thread_sum, 1, 2);

    if (tid == 0) {
        output[n * spatial_dim + s] = convert_float_value<T>(log(thread_sum) + smax[0]);
    }
}

Status CudaReduceLogSumExpLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaReduceLogSumExpLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaReduceLogSumExpLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<ReduceLayerParam *>(param_);
    if (!params) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];

    int channels = 1;
    int first_axis = 4;
    int last_axis = 0;
    // remove duplicate axes
    auto axis = params->axis;
    std::sort(axis.begin(), axis.end());
    axis.erase(std::unique(axis.begin(), axis.end() ), axis.end());
    for (int i = 0; i < axis.size(); i++) {
        channels *= input_blob->GetBlobDesc().dims[axis[i]];
        first_axis = std::min(axis[i], first_axis);
        last_axis = std::max(axis[i], last_axis);
    }

    for(int i=first_axis; i<=last_axis; ++i) {
        if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
            LOGE("Error: discontinuous reduce axes!");
            return Status(TNNERR_PARAM_ERR, "Error: discontinuous reduce axes!"); 
        }
    }

    int outer_dim = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims, 0, first_axis);
    int inner_dim = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims, last_axis+1);
    int count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    const int BLOCKSIZE = 128;
    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float* input_data = static_cast<float*>(input_blob->GetHandle().base);
        float* output_data = static_cast<float*>(output_blob->GetHandle().base);
        reduce_log_sum_exp_kernel<BLOCKSIZE, float><<<count, BLOCKSIZE, BLOCKSIZE*sizeof(float), context_->GetStream()>>>(
            outer_dim, channels, inner_dim, input_data, output_data);
    } else if (input_blob->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        __half* input_data = static_cast<__half*>(input_blob->GetHandle().base);
        __half* output_data = static_cast<__half*>(output_blob->GetHandle().base);
        reduce_log_sum_exp_kernel<BLOCKSIZE, __half><<<count, BLOCKSIZE, BLOCKSIZE*sizeof(float), context_->GetStream()>>>(
            outer_dim, channels, inner_dim, input_data, output_data);
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc don't support datatype");
    }
    return TNN_OK;
}

REGISTER_CUDA_ACC(ReduceLogSumExp, LAYER_REDUCE_LOG_SUM_EXP);

}  // namespace TNN_NS
