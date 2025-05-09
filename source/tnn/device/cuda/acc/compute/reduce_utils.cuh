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

#ifndef TNN_CUDA_REDUCE_UTILS_H_
#define TNN_CUDA_REDUCE_UTILS_H_

namespace TNN_NS {

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    return val;
}

template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int lane_id = threadIdx.x & 0x1f;
    int warp_id = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane_id == 0)
        shared[warp_id] = val;
    __syncthreads();

    if (threadIdx.x < 32) {
        val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[threadIdx.x] : 0;
        val = warpReduceSum(val);
    }
    return val;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    return val;
}

// Calculate the maximum of all elements in a block
template<typename T>
__inline__ __device__ T blockReduceMax(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;  // in-warp idx
    int wid = threadIdx.x >> 5;     // warp idx

    val = warpReduceMax(val);  // get maxx in each warp
    if (lane == 0)  // record in-warp maxx by warp Idx
        shared[wid] = val;
    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}



template<typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val) {
    #pragma unroll
    for (int i = 0; i < NUM; i++) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(0xffffffff, val[i], mask, 32);
    }
    return (T)(0.0f);
}

template<typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
    static __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSumV2<T, NUM>(val);

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < NUM; i++) {
            shared[i][wid] = val[i];
        }
    }
    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
    #pragma unroll
    for (int i = 0; i < NUM; i++) {
        val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
    }
    warpReduceSumV2<T, NUM>(val);
    return (T)0.0f;
}

template<typename T, int NUM>
__inline__ __device__ T warpReduceMaxV2(T* val) {
    #pragma unroll
    for (int i = 0; i < NUM; i++) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] = max(val[i], __shfl_xor_sync(0xffffffff, val[i], mask, 32));
    }
    return (T)(0.0f);
}

template<typename T, int NUM>
__inline__ __device__ T blockReduceMaxV2(T* val) {
    static __shared__ T shared[32][NUM];
    int lane = threadIdx.x & 0x1f;  // in-warp idx
    int wid = threadIdx.x >> 5;     // warp idx

    warpReduceMaxV2<T, NUM>(val);  // get maxx in each warp

    if (lane == 0) { // record in-warp maxx by warp Idx
        #pragma unroll
        for (int i = 0; i < NUM; i++) {
            shared[wid][i] = val[i];
        }
    }
    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
    #pragma unroll
    for (int i = 0; i < NUM; i++) {
        val[i] = is_mask ? shared[lane][i] : (T)-1e20f;
    }
    warpReduceMaxV2<T, NUM>(val);
    return (T)0.0f;
}

}   // namespace TNN_NS

#endif  // TNN_CUDA_REDUCE_UTILS_H_
