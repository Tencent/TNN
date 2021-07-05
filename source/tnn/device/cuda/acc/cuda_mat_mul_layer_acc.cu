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
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(MatMul, LAYER_MATMUL);

#define BLOCK_DIM 16

__device__ __forceinline__ __half atomic_add(__half* address, __half val) {
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
    return atomicAdd(address, val);
#else
    unsigned int* address_as_uint = (unsigned int*) address;
    unsigned int old = *address_as_uint;
    __half* old_as_half = (__half*) &old;
    unsigned int assumed;
    unsigned int updated;
    __half* updated_as_half = (__half*) &updated;
    do {
        assumed = old;
        updated = old;
        *updated_as_half = __hadd(val, *updated_as_half);
        old = atomicCAS(address_as_uint, assumed, updated);
    } while (assumed != old);
    return *old_as_half;
#endif // __CUDA_ARCH__ >= 700
}

template<typename T>
__global__ void matmul_transpose_kernel(T *odata, T *idata, int width, int height) {
    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

    odata += blockIdx.z * width * height;
    idata += blockIdx.z * width * height;
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if((xIndex < width) && (yIndex < height)) {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if((xIndex < height) && (yIndex < width)) {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

__global__ void matmul_batched_gemv_kernel(const float* data1, const float* data2, float* output,
        int stride_a1, int stride_a2, int stride_a3, int stride_b1, int stride_b2, int stride_b3,
        int size2, int size3, int N, int K) {

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

    const float* a = data1 + offset_a * N * K + blockIdx.y * blockDim.x + blockIdx.z * TNN_CUDA_NUM_THREADS * N;
    const float* b = data2 + offset_b * K;
    float* out = output + offset_out * N + blockIdx.y * blockDim.x;

    int group = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    float value_b = threadIdx.x + blockIdx.z * TNN_CUDA_NUM_THREADS < K ?
        b[threadIdx.x + blockIdx.z * TNN_CUDA_NUM_THREADS] : 0;

    a += group * 32 * N;
    float local_sum[4] = {0, 0, 0, 0};

    int end = max(0, min(32, K - group * 32));

    for (int j = 0; j < end; j++) {
        float bx = __shfl_sync(0xffffffff, value_b, j, 32);
        for (int i = 0; i < 4; i++) {
            int new_group = (group + i) % 4;
            int offset = new_group * 32 + lane;
            if (blockIdx.y * blockDim.x + offset < N) {
                local_sum[i] = __fmaf_rn(a[offset + j * N], bx, local_sum[i]);
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        int new_group = (group + i) % 4;
        int offset = new_group * 32 + lane;
        if (blockIdx.y * blockDim.x + offset < N)
            atomicAdd(&out[offset], local_sum[i]);
    }
}

__global__ void matmul_batched_gemv_kernel_fp16(const __half* data1, const float* data2, __half * output,
        int stride_a1, int stride_a2, int stride_a3, int stride_b1, int stride_b2, int stride_b3,
        int size2, int size3, int N, int K) {

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

    const __half* a = data1 + offset_a * N * K + blockIdx.y * blockDim.x + blockIdx.z * TNN_CUDA_NUM_THREADS * N;
    const float* b = data2 + offset_b * K;
    __half * out = output + offset_out * N + blockIdx.y * blockDim.x;

    int group = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    float value_b = threadIdx.x + blockIdx.z * TNN_CUDA_NUM_THREADS < K ?
        b[threadIdx.x + blockIdx.z * TNN_CUDA_NUM_THREADS] : 0.f;

    a += group * 32 * N;
    float local_sum[4] = {0, 0, 0, 0};

    int end = max(0, min(32, K - group * 32));

    for (int j = 0; j < end; j++) {
        float bx = __shfl_sync(0xffffffff, value_b, j, 32);
        for (int i = 0; i < 4; i++) {
            int new_group = (group + i) % 4;
            int offset = new_group * 32 + lane;
            if (blockIdx.y * blockDim.x + offset < N) {
                local_sum[i] = __fmaf_rn(__half2float(a[offset + j * N]), bx, local_sum[i]);
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        int new_group = (group + i) % 4;
        int offset = new_group * 32 + lane;
        if (blockIdx.y * blockDim.x + offset < N)
            atomic_add(&out[offset], __float2half(local_sum[i]));
    }
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
        LOGE("Error: layer acc dont support dims: %lu\n", input_dims1.size());
        return Status(TNNERR_MODEL_ERR, "Error: layer acc don't support datatype");
    }

    int K = input_dims1[input_dims1.size() - 1];
    int N = input_dims1[input_dims1.size() - 2];

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

    void* input_data1 = input_blob1->GetHandle().base;
    void* input_data2 = input_blob2->GetHandle().base;
    void* output_data = output_blob->GetHandle().base;

    dim3 dimGrid(K/BLOCK_DIM, N/BLOCK_DIM, size[0]*stride_a[0]+size[1]*stride_a[1]+size[2]*stride_a[2]);
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);

    int type_size = DataTypeUtils::GetBytesSize(input_blob1->GetBlobDesc().data_type);
    int cur_workspace_size = (size[0]*stride_a[0]+size[1]*stride_a[1]+size[2]*stride_a[2]) * K * N * type_size;

    context_->SetWorkspaceSize(cur_workspace_size);
    if (input_blob1->GetBlobDesc().data_type == DataType::DATA_TYPE_FLOAT) {
        matmul_transpose_kernel<<<dimGrid, dimBlock, 0, context_->GetStream()>>>((float*)context_->GetWorkspace(),
        (float*)input_data1, K, N);
    } else if (input_blob1->GetBlobDesc().data_type == DataType::DATA_TYPE_HALF) {
        matmul_transpose_kernel<<<dimGrid, dimBlock, 0, context_->GetStream()>>>((__half*)context_->GetWorkspace(),
        (__half*)input_data1, K, N);
    }

    dim3 grid;
    grid.x = size[0] * size[1] * size[2];
    grid.y = (N + TNN_CUDA_NUM_THREADS - 1) / TNN_CUDA_NUM_THREADS;
    grid.z = (K + TNN_CUDA_NUM_THREADS - 1) / TNN_CUDA_NUM_THREADS;

    CUDA_CHECK(cudaMemsetAsync(output_data, 0, size[0] * size[1] * size[2] * N * type_size, context_->GetStream()));

    if (input_blob1->GetBlobDesc().data_type == DataType::DATA_TYPE_FLOAT) {
        matmul_batched_gemv_kernel<<<grid, TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            (float*)context_->GetWorkspace(), (float*)input_data2, (float*)output_data, stride_a[0], stride_a[1],
            stride_a[2], stride_b[0], stride_b[1], stride_b[2], size[1], size[2], N, K);
    } else if (input_blob1->GetBlobDesc().data_type == DataType::DATA_TYPE_HALF) {
        matmul_batched_gemv_kernel_fp16<<<grid, TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            (__half*)context_->GetWorkspace(), (float*)input_data2, (__half*)output_data, stride_a[0], stride_a[1],
            stride_a[2], stride_b[0], stride_b[1], stride_b[2], size[1], size[2], N, K);
    }
    return TNN_OK;
}

REGISTER_CUDA_ACC(MatMul, LAYER_MATMUL);

}  // namespace TNN_NS
