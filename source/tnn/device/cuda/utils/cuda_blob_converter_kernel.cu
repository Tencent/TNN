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

#include "tnn/device/cuda/utils/cuda_blob_converter_kernel.cuh"

namespace TNN_NS {

#define ELEMENT_PER_THREAD 4
#define THREAD_PER_BLOCK 128

inline __device__ unsigned char fp32_to_u8_sat(float in) {
    int x = __float2int_rn(in);
    x = x > 255 ? 255 : x;
    x = x > 0 ? x : 0;
    return (unsigned char)(x);
}

__global__ void blob_to_bgr_kernel(int CHW, int HW, const float* __restrict__ src, unsigned char *dst,
        int channels, float *scale, float *bias) {
    const int offset = ELEMENT_PER_THREAD * THREAD_PER_BLOCK * blockIdx.x + threadIdx.x;

    src += offset + blockIdx.y * CHW;
    dst += offset * channels + blockIdx.y * CHW;
    int channels_coef = channels - 1;

    #pragma unroll
    for (int c = 0; c < channels; ++c) {
        unsigned char data_ld[ELEMENT_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ELEMENT_PER_THREAD; ++i) {
            if (i * THREAD_PER_BLOCK + offset < HW) {
                data_ld[i] = fp32_to_u8_sat(src[i * THREAD_PER_BLOCK + (channels_coef - c) * HW]
                                                * scale[channels_coef - c]
                                                + bias[channels_coef - c]);
            }
        }
        #pragma unroll
        for (int i = 0; i < ELEMENT_PER_THREAD; ++i) {
            if (i * THREAD_PER_BLOCK + offset < HW) {
                dst[c + i * THREAD_PER_BLOCK * channels] = data_ld[i];
            }
        }
    }
}

__global__ void blob_to_gray_kernel(int count, const float *src, unsigned char *dst, float scale, float bias) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        dst[index] = fp32_to_u8_sat(scale * src[index] + bias);
}

__global__ void bgr_to_blob_kernel(int CHW, int HW, const unsigned char* __restrict__ src, float *dst,
        int channels, float *scale, float *bias) {
    const int offset = ELEMENT_PER_THREAD * THREAD_PER_BLOCK * blockIdx.x + threadIdx.x;

    src += offset * channels + blockIdx.y * CHW;
    dst += offset + blockIdx.y * CHW;

    #pragma unroll
    for (int c = 0; c < channels; ++c) {
        float data_ld[ELEMENT_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ELEMENT_PER_THREAD; ++i) {
            if (i * THREAD_PER_BLOCK + offset < HW) {
                data_ld[i] = (src[i * THREAD_PER_BLOCK * channels + c] * scale[c] + bias[c]);
            }
        }
        #pragma unroll
        for (int i = 0; i < ELEMENT_PER_THREAD; ++i) {
            if (i * THREAD_PER_BLOCK + offset < HW) {
                dst[c * HW + i * THREAD_PER_BLOCK] = data_ld[i];
            }
        }
    }
}

__global__ void gray_to_blob_kernel(int count, const unsigned char *src, float *dst, float scale, float bias) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        dst[index] = scale * src[index] + bias;
}

void BlobToBGR(int batch, int CHW, int HW, const float *src, unsigned char *dst, cudaStream_t stream,
        int channels, float *scale, float *bias) {
    dim3 grid;
    grid.x = (HW + ELEMENT_PER_THREAD * THREAD_PER_BLOCK - 1) / (ELEMENT_PER_THREAD * THREAD_PER_BLOCK);
    grid.y = batch;
    blob_to_bgr_kernel<<<grid, THREAD_PER_BLOCK, 0, stream>>>(
        CHW, HW, src, dst, channels, scale, bias);
}

void BlobToGray(int count, const float *src, unsigned char *dst, cudaStream_t stream, float scale, float bias) {
    const int BLOCK_NUM = (count + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    blob_to_gray_kernel<<<BLOCK_NUM, THREAD_PER_BLOCK, 0, stream>>>(count, src, dst, scale, bias);
}

void BGRToBlob(int batch, int CHW, int HW, const unsigned char *src, float *dst, cudaStream_t stream,
        int channels, float *scale, float* bias) {
    dim3 grid;
    grid.x = (HW + ELEMENT_PER_THREAD * THREAD_PER_BLOCK - 1) / (ELEMENT_PER_THREAD * THREAD_PER_BLOCK);
    grid.y = batch;
    bgr_to_blob_kernel<<<grid, THREAD_PER_BLOCK, 0, stream>>>(
        CHW, HW, src, dst, channels, scale, bias);
}

void GrayToBlob(int count, const unsigned char *src, float *dst, cudaStream_t stream, float scale, float bias) {
    const int BLOCK_NUM = (count + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    gray_to_blob_kernel<<<BLOCK_NUM, THREAD_PER_BLOCK, 0, stream>>>(count, src, dst, scale, bias);
}

}  //  namespace TNN_NS
