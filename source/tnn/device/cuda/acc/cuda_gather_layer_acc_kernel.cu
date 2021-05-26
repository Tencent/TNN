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

#include "tnn/device/cuda/acc/cuda_gather_layer_acc_kernel.cuh"

namespace TNN_NS {

template <typename T>
__global__ void gather_kernel(
    int dst_size,
    int slice_size,
    int src_slice_count,
    int dst_slice_count,
    const T* src_data,
    const int* indices_data,
    T* dst_data) {

    int dst_batch_size = dst_slice_count * slice_size;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dst_size) {
        int batch_idx = tid / dst_batch_size;
        int dst_idx_per_batch = tid % dst_batch_size;
        int dst_slice_idx = dst_idx_per_batch / slice_size;
        int offset_per_slice = dst_idx_per_batch % slice_size;

        int src_slice_idx = indices_data[dst_slice_idx];
        src_slice_idx = src_slice_idx < 0 ?
                        src_slice_idx + src_slice_count : src_slice_idx;

        if (src_slice_idx < 0 || src_slice_idx >= src_slice_count) {
            dst_data[tid] = 0;
            return;
        }

        int src_idx = batch_idx * src_slice_count * slice_size;
        src_idx += src_slice_idx * slice_size + offset_per_slice;
        dst_data[tid] = src_data[src_idx];
    }
}

template <typename T>
Status RunGather(int dst_size, int slice_size, int src_slice_count, int dst_slice_count,
                 const T* src_data, const int* indices_data, T* dst_data, cudaStream_t stream) {
    const int THREAD_PER_BLOCK = 128;
    int blocks = (dst_size + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    gather_kernel<T><<<blocks, THREAD_PER_BLOCK, 0, stream>>>(
        dst_size, slice_size, src_slice_count, dst_slice_count,
        src_data, indices_data, dst_data);

    return TNN_OK;
}

template Status RunGather<float>(int dst_size, int slice_size, int src_slice_count, int dst_slice_count,
                 const float* src_data, const int* indices_data, float* dst_data, cudaStream_t stream);
template Status RunGather<int>(int dst_size, int slice_size, int src_slice_count, int dst_slice_count,
                 const int* src_data, const int* indices_data, int* dst_data, cudaStream_t stream);
template Status RunGather<__half>(int dst_size, int slice_size, int src_slice_count, int dst_slice_count,
                 const __half* src_data, const int* indices_data, __half* dst_data, cudaStream_t stream);

}
