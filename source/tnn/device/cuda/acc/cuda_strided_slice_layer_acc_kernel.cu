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

#include "tnn/device/cuda/acc/cuda_strided_slice_layer_acc_kernel.cuh"

namespace TNN_NS {

template<int THREAD_PER_BLOCK, int ELE_PER_THREAD>
__global__ void strided_slice_kernel(int size, const float * __restrict__ srcData, int input_c, int input_h,
        int input_w, const int* __restrict__ begin, const int* __restrict__ strides, float* __restrict__ dstData,
        int output_c, int output_h, int output_w, int div_c, int div_n) {
    int block_offset = blockIdx.x * THREAD_PER_BLOCK * ELE_PER_THREAD;

    const int mul_n = input_c * input_h * input_w * strides[3];
    const int mul_c = input_h * input_w * strides[2];
    const int mul_h = input_w * strides[1];
    const int mul_w = strides[0];
    const int offset = begin[3] * input_c * input_h * input_w +
                   + begin[2] * input_h * input_w +
                   + begin[1] * input_w
                   + begin[0];

    #pragma unroll
    for(int i =0;i < ELE_PER_THREAD ;i++) {
        int index = block_offset + i * THREAD_PER_BLOCK + threadIdx.x;
        if (index < size) {
            int w = index % output_w;
            int h = index / output_w % output_h;
            int c = index / div_c % output_c;
            int n = index / div_n ;
            int input_index = n * mul_n + c * mul_c + h * mul_h + w * mul_w + offset;
            dstData[index] = srcData[input_index];
        }
    }
}

template<int THREAD_PER_BLOCK, int ELE_PER_THREAD>
__global__ void strided_slice_v2_kernel(int size, const float * __restrict__ srcData, int input_c, int input_d, int input_h,
        int input_w, const int* __restrict__ begin, const int* __restrict__ strides, float* __restrict__ dstData,
        int output_c, int output_d, int output_h, int output_w, int div_d, int div_c, int div_n) {
    int block_offset = blockIdx.x * THREAD_PER_BLOCK * ELE_PER_THREAD;

    const int mul_n = input_c * input_d * input_h * input_w * strides[4];
    const int mul_c = input_d * input_h * input_w * strides[3];
    const int mul_d = input_h * input_w * strides[2];
    const int mul_h = input_w * strides[1];
    const int mul_w = strides[0];
    const int offset = begin[4] * input_c * input_d * input_h * input_w
                   + begin[3] * input_d * input_h * input_w
                   + begin[2] * input_h * input_w +
                   + begin[1] * input_w
                   + begin[0];

    #pragma unroll
    for(int i =0;i < ELE_PER_THREAD ;i++) {
        int index = block_offset + i * THREAD_PER_BLOCK + threadIdx.x;
        if (index < size) {
            int w = index % output_w;
            int h = index / output_w % output_h;
            int d = index / div_d % output_d;
            int c = index / div_c % output_c;
            int n = index / div_n ;
            int input_index = n * mul_n + c * mul_c + d * mul_d + h * mul_h + w * mul_w + offset;
            dstData[index] = srcData[input_index];
        }
    }
}

Status RunStrideSlice(int size, const float * src_data, int input_c, int input_h,
        int input_w, const int* begin, const int* strides, float* dst_data,
        int output_c, int output_h, int output_w, int div_c, int div_n, cudaStream_t stream) {
     
    const int THREAD_PER_BLOCK = 128;
    const int ELE_PER_THREAD = 64;
    int blocks = (size + THREAD_PER_BLOCK * ELE_PER_THREAD - 1) / (THREAD_PER_BLOCK * ELE_PER_THREAD);
    strided_slice_kernel<THREAD_PER_BLOCK, ELE_PER_THREAD><<<blocks, THREAD_PER_BLOCK, 0, stream>>>(
        size, src_data, input_c, input_h, input_w, begin, strides,
        dst_data, output_c, output_h, output_w, div_c, div_n);
    return TNN_OK;
}

Status RunStrideSlice(int size, const float * src_data, int input_c, int input_d, int input_h,
        int input_w, const int* begin, const int* strides, float* dst_data,
        int output_c, int output_d, int output_h, int output_w, int div_d, int div_c, int div_n, cudaStream_t stream) {
     
    const int THREAD_PER_BLOCK = 128;
    const int ELE_PER_THREAD = 64;
    int blocks = (size + THREAD_PER_BLOCK * ELE_PER_THREAD - 1) / (THREAD_PER_BLOCK * ELE_PER_THREAD);
    strided_slice_v2_kernel<THREAD_PER_BLOCK, ELE_PER_THREAD><<<blocks, THREAD_PER_BLOCK, 0, stream>>>(
        size, src_data, input_c, input_d, input_h, input_w, begin, strides,
        dst_data, output_c, output_d, output_h, output_w, div_d, div_c, div_n);
    return TNN_OK;
}


}
