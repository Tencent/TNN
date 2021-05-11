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

DECLARE_CUDA_ACC(MatMul, LAYER_MATMUL);

/*
__global__ void matmul_batched_gemv_kernel(const float* data1, const float* data2, float* output,
        int stride_a1, int stride_a2, int stride_a3, int stride_b1, int stride_b2, int stride_b3,
        int size2, int size3, int N, int K) {
    __shared__ float sm[256 * 8];

    int index1 = blockIdx.x / (size2 * size3);
    int index2 = blockIdx.x / size3 % size2;
    int index3 = blockIdx.x % size3;

    int offset_a = index1 * stride_a1 * (size2 * size3) +
                index2 * stride_a2 * size3 +
                index3 * stride_a3;

    int offset_b = index1 * stride_b1 * (size2 * size3) +
                index2 * stride_b2 * size3 +
                index3 * stride_b3;

    int offset_out = index1 * (size2 * size3) +
                index2 * size3 + index3;

    const float* a = data1 + offset_a * N * K;
    const float* b = data2 + offset_b * K;
    float* out = output + offset_out * N;
    int n_offset = blockIdx.y * 256;
    int kid = threadIdx.x + blockIdx.z * blockDim.x;

    float sum = 0;
    if (K % 4 == 0 && kid < K/4) {
        float4 value_b = ((const float4*)b)[kid];
        for (int j = 0; n_offset + j < N && j < 256; j++) {
            float4 value_a = ((const float4*)a)[(n_offset + j) * K/4 + kid];
            sum = 0;
            sum = __fmaf_rn(value_a.x, value_b.x, sum);
            sum = __fmaf_rn(value_a.y, value_b.y, sum);
            sum = __fmaf_rn(value_a.z, value_b.z, sum);
            sum = __fmaf_rn(value_a.w, value_b.w, sum);

            sum += __shfl_down_sync(0xffffffff, sum, 16, 32);
            sum += __shfl_down_sync(0x0000ffff, sum, 8, 16);
            sum += __shfl_down_sync(0x000000ff, sum, 4, 8);
            sum += __shfl_down_sync(0x0000000f, sum, 2, 4);
            sum += __shfl_down_sync(0x00000003, sum, 1, 2);

            if (threadIdx.x % 32 == 0) {
                int line = threadIdx.x / 128;
                int lane = threadIdx.x / 32 % 4;
                sm[256 * line * 4 + j * 4 + lane] = sum;
            }
        }
        __syncthreads();

        if (n_offset + threadIdx.x < N) {
            float tmp = 0;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                float4 result = ((float4*)sm)[threadIdx.x + i * 256];
                tmp += result.x + result.y + result.z + result.w;
            }
            out[n_offset + threadIdx.x] = tmp;
        }
    } else if (K % 4 != 0 && kid < K) {
        float value_b = b[kid];
        for (int j = 0; n_offset + j < N && j < 256; j++) {
            float value_a = a[(n_offset + j) * K + kid];
            sum = 0;
            sum = __fmaf_rn(value_a, value_b, sum);
            sum += __shfl_down_sync(0xffffffff, sum, 16, 32);
            sum += __shfl_down_sync(0x0000ffff, sum, 8, 16);
            sum += __shfl_down_sync(0x000000ff, sum, 4, 8);
            sum += __shfl_down_sync(0x0000000f, sum, 2, 4);
            sum += __shfl_down_sync(0x00000003, sum, 1, 2);
            if (threadIdx.x % 32 == 0) {
                int line = threadIdx.x / 128;
                int lane = threadIdx.x / 32 % 4;
                sm[256 * line * 4 + j * 4 + lane] = sum;
            }
        }
        __syncthreads();

        if (n_offset + threadIdx.x < N) {
            float tmp = 0;
            for (int i = 0; i < 2; i++) {
                float4 result = ((float4*)sm)[threadIdx.x + i * 256];
                tmp += result.x + result.y + result.z + result.w;
            }
            atomicAdd(&out[n_offset + threadIdx.x], tmp);
        }
    }
}

__global__ void matmul_batched_gemv_kernel(const float* data1, const float* data2, float* output,
        int stride_a1, int stride_a2, int stride_a3, int stride_b1, int stride_b2, int stride_b3,
        int size2, int size3, int N, int K) {
    extern __shared__ float sm[];

    int index1 = blockIdx.x / (size2 * size3);
    int index2 = blockIdx.x / size3 % size2;
    int index3 = blockIdx.x % size3;

    int offset_a = index1 * stride_a1 * (size2 * size3) +
                index2 * stride_a2 * size3 +
                index3 * stride_a3;

    int offset_b = index1 * stride_b1 * (size2 * size3) +
                index2 * stride_b2 * size3 +
                index3 * stride_b3;

    int offset_out = index1 * (size2 * size3) +
                index2 * size3 + index3;

    const float* a = data1 + offset_a * N * K;
    const float* b = data2 + offset_b * K;
    float* out = output + offset_out * N;
    int n_offset = blockIdx.y * 128;
    a += n_offset * K;
    int nid = threadIdx.x + blockIdx.y * blockDim.x;

    for (int i = threadIdx.x; i < K/4; i+=blockDim.x) {
        ((float4*)sm)[i] = ((const float4*)b)[i];
    }

    float* sm_a = &sm[1024];

    int width = 8;
    int height = blockDim.x / width;
    int row = threadIdx.x / width;
    int col = threadIdx.x % width;

    float sum = 0;
    for (int i = 0; i < K/4 / width; i++) {
        __syncthreads();
        for (int j = 0; j * height + row + n_offset < N && j < width; j++) {
            //if (i * width + col < K/4) {
                ((float4*)sm_a)[blockDim.x * col + j * height + row] = ((const float4*)a)[(j * height + row) * K/4 + col];
            //}
        }
        __syncthreads();

        if (nid < N) {
            for (int j = 0; j < width; j++) {
                //if (i * width + j < K/4) {
                    float4 value_a = ((float4*)sm_a)[j * blockDim.x + threadIdx.x];
                    float4 value_b = ((float4*)sm)[i * width + j];
                    sum += value_a.x * value_b.x;
                    sum += value_a.y * value_b.y;
                    sum += value_a.z * value_b.z;
                    sum += value_a.w * value_b.w;
                //}
            }
        }
        a += width * 4;
    }

    if (nid < N)
        out[nid] = sum;
}
*/

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeNoBankConflicts(float *odata, const float *idata) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    odata += blockIdx.z * 1024 * 1024;
    idata += blockIdx.z * 1024 * 1024;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

/*
__global__ void matmul_batched_gemv_kernel(const float* data1, const float* data2, float* output,
        int stride_a1, int stride_a2, int stride_a3, int stride_b1, int stride_b2, int stride_b3,
        int size2, int size3, int N, int K) {
    extern __shared__ float sm[];

    int index1 = blockIdx.x / (size2 * size3);
    int index2 = blockIdx.x / size3 % size2;
    int index3 = blockIdx.x % size3;

    int offset_a = index1 * stride_a1 * (size2 * size3) +
                index2 * stride_a2 * size3 +
                index3 * stride_a3;

    int offset_b = index1 * stride_b1 * (size2 * size3) +
                index2 * stride_b2 * size3 +
                index3 * stride_b3;

    int offset_out = index1 * (size2 * size3) +
                index2 * size3 + index3;

    const float* a = data1 + offset_a * N * K + blockIdx.y * blockDim.x;
    const float* b = data2 + offset_b * K;
    float* out = output + offset_out * N + blockIdx.y * blockDim.x;

    float sum = 0;
    for (int i = threadIdx.x; i < K/4; i += blockDim.x) {
        ((float4*)sm)[i] = ((const float4*)b)[i];
    }

    __syncthreads();

    for (int i = 0; i < K/4; i++) {
        float4 value_b = ((float4*)sm)[i];
        sum = __fmaf_rn(a[threadIdx.x + (i * 4 + 0) * N], value_b.x, sum);
        sum = __fmaf_rn(a[threadIdx.x + (i * 4 + 1) * N], value_b.y, sum);
        sum = __fmaf_rn(a[threadIdx.x + (i * 4 + 2) * N], value_b.z, sum);
        sum = __fmaf_rn(a[threadIdx.x + (i * 4 + 3) * N], value_b.w, sum);
    }
    out[threadIdx.x] = sum;
}
*/
__global__ void matmul_batched_gemv_kernel(const float* data1, const float* data2, float* output,
        int stride_a1, int stride_a2, int stride_a3, int stride_b1, int stride_b2, int stride_b3,
        int size2, int size3, int N, int K) {
    __shared__ float sm[128];

    int index1 = blockIdx.x / (size2 * size3);
    int index2 = blockIdx.x / size3 % size2;
    int index3 = blockIdx.x % size3;

    int offset_a = index1 * stride_a1 * (size2 * size3) +
                index2 * stride_a2 * size3 +
                index3 * stride_a3;

    int offset_b = index1 * stride_b1 * (size2 * size3) +
                index2 * stride_b2 * size3 +
                index3 * stride_b3;

    int offset_out = index1 * (size2 * size3) +
                index2 * size3 + index3;

    const float* a = data1 + offset_a * N * K + blockIdx.y * blockDim.x + blockIdx.z * 128 * N;
    const float* b = data2 + offset_b * K;
    float* out = output + offset_out * N + blockIdx.y * blockDim.x;

    int group = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    float value_b = ((const float*)b)[threadIdx.x + blockIdx.z * 128];

    sm[threadIdx.x] = 0;

    __syncthreads();

    a += group * 32 * N;
    float local_sum[4] = {0};

    for (int j = 0; j < 32; j++) {
        float bx = __shfl_sync(0xffffffff, value_b, j, 32);
//        float by = __shfl_sync(0xffffffff, value_b.y, j, 32);
//        float bz = __shfl_sync(0xffffffff, value_b.z, j, 32);
//        float bw = __shfl_sync(0xffffffff, value_b.w, j, 32);
        for (int i = 0; i < 4; i++) {
            int new_group = (group + i) % 4;
            int offset = new_group * 32 + lane;
            local_sum[i] = __fmaf_rn(a[offset + (j * 1 + 0) * N], bx, local_sum[i]);
//            local_sum[i] = __fmaf_rn(a[offset + (j * 4 + 1) * N], by, local_sum[i]);
//            local_sum[i] = __fmaf_rn(a[offset + (j * 4 + 2) * N], bz, local_sum[i]);
//            local_sum[i] = __fmaf_rn(a[offset + (j * 4 + 3) * N], bw, local_sum[i]);
            //sum = __fmaf_rn(a[offset + (j * 4 + 1) * N], by, sum);
            //sum = __fmaf_rn(a[offset + (j * 4 + 2) * N], bz, sum);
            //sum = __fmaf_rn(a[offset + (j * 4 + 3) * N], bw, sum);
//            sm[offset] += sum;
//            __syncthreads();
        }
    }

    for (int i = 0; i < 4; i++) {
        int new_group = (group + i) % 4;
        int offset = new_group * 32 + lane;
        atomicAdd(&sm[offset], local_sum[i]);
    }

    __syncthreads();
    atomicAdd(&out[threadIdx.x], sm[threadIdx.x]);
}

Status CudaMatMulLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaMatMulLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaMatMulLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob* input_blob1 = inputs[0];
    Blob* input_blob2 = inputs[1];
    Blob* output_blob = outputs[0];
    auto input_dims1 = input_blob1->GetBlobDesc().dims;
    auto input_dims2 = input_blob2->GetBlobDesc().dims;

    if (input_dims1.size() > 5) {
        LOGE("Error: layer acc dont support dims: %d\n", input_dims1.size());
        return Status(TNNERR_MODEL_ERR, "Error: layer acc don't support datatype");
    }

    int K = input_dims1[input_dims1.size() - 2];
    int N = input_dims1[input_dims1.size() - 1];

    int size[3];
    int stride_a[3];
    int stride_b[3];

    int i = 0;
    for (; i < input_dims1.size() - 2; i++) {
        size[i] = std::max(input_dims1[i], input_dims2[i]);
        stride_a[i] = input_dims1[i] == 1 ? 0 : 1;
        stride_b[i] = input_dims2[i] == 1 ? 0 : 1;
    }

    for (; i < 3; i++) {
        size[i] = 1;
        stride_a[i] = 0;
        stride_b[i] = 0;
    }

    float* input_data1 = static_cast<float*>(input_blob1->GetHandle().base);
    float* input_data2 = static_cast<float*>(input_blob2->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);

    dim3 dimGrid(1024/TILE_DIM, 1024/TILE_DIM, size[0]*stride_a[0]+size[1]*stride_a[1]+size[2]*stride_a[2]);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    transposeNoBankConflicts<<<dimGrid, dimBlock, 0, context_->GetStream()>>>((float*)tmp_data, input_data1);

    dim3 grid;
    grid.x = size[0] * size[1] * size[2];
    grid.y = (N + 127) / 128;
/*    if (K % 4 == 0) {
        grid.z = (K/4 + 255) / 256;
    } else {
        grid.z = (K + 255) / 256;
    }
*/
    grid.z = (K + 127) / 128;

    cudaMemsetAsync(output_data, 0, size[0] * size[1] * size[2] * N * sizeof(float), context_->GetStream());
    matmul_batched_gemv_kernel<<<grid, 128, 0, context_->GetStream()>>>(
        (float*)tmp_data, input_data2, output_data, stride_a[0], stride_a[1], stride_a[2], stride_b[0],
        stride_b[1], stride_b[2], size[1], size[2], N, K);

    return TNN_OK;
}

REGISTER_CUDA_ACC(MatMul, LAYER_MATMUL);

}  // namespace TNN_NS
