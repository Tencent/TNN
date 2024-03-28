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

#include "tnn/device/cuda/utils.cuh"
#include "tnn/device/cuda/acc/cuda_mat_mul_layer_acc.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_type_utils.h"
//#include <cublas_v2.h>

namespace TNN_NS {

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
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }

    auto mm_param = dynamic_cast<MatMulLayerParam *>(param);
    if (!mm_param) {
        LOGE("Error: Unable to Get Param of CUDA MatMul Layer.\n");
        return Status(TNNERR_LAYER_ERR, "Error: Unable to Get Param of CUDA MatMul Layer.");
    }

    DataType compute_dtype = inputs[0]->GetBlobDesc().data_type;
    if (compute_dtype != DATA_TYPE_FLOAT && compute_dtype != DATA_TYPE_HALF) {
        LOGE("Error: MatMul input Mat has data type other than float and half, which is Not Supported by TNN cuda ACC.\n");
        return Status(TNNERR_MODEL_ERR, "Error: MatMul input Mat has data type other than float and half, which is Not Supported by TNN cuda ACC.");
    }

    if (mm_param->extra_config.find("ffn") != mm_param->extra_config.end()) {
        if (inputs.size() == 1) {
            if (mm_param->weight_position != 0 && mm_param->weight_position != 1) {
                LOGE("Error: Wrong layer param for CUDA MatMul Layer.\n");
                return Status(TNNERR_LAYER_ERR, "Error: Wrong layer param for CUDA MatMul Layer.");
            }
            auto mm_resource = dynamic_cast<MatMulLayerResource *>(resource);
            if (!mm_resource) {
                LOGE("Error: Unable to Get Resource of CUDA MatMul Layer.\n");
                return Status(TNNERR_LAYER_ERR, "Error: Unable to Get Resource of CUDA MatMul Layer.");
            }
            RawBuffer buf = mm_resource->weight;
            if (buf.GetDataCount() <= 0 ||
                (buf.GetDataType() != DATA_TYPE_FLOAT && buf.GetDataType() != DATA_TYPE_HALF)) {
                LOGE("Error: Unable to Get Correct Param and Resource of CUDA MatMul Layer.\n");
                return Status(TNNERR_LAYER_ERR, "Error: Unable to Get Correct Param and Resource of CUDA MatMul Layer.");
            }

            CreateTempBuf(buf.GetDataCount() * DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT));
            CreateTempBuf(buf.GetDataCount() * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));

            if (buf.GetDataType() == DATA_TYPE_FLOAT) {
                CUDA_CHECK(cudaMemcpy(tempbufs_[0].ptr,
                                      buf.force_to<void*>(),
                                      buf.GetDataCount() * DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT),
                                      cudaMemcpyHostToDevice));
                auto half_buf = ConvertFloatToHalf(buf);
                CUDA_CHECK(cudaMemcpy(tempbufs_[1].ptr,
                                      half_buf.force_to<void*>(),
                                      half_buf.GetDataCount() * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF),
                                      cudaMemcpyHostToDevice));
            } else if (buf.GetDataType() == DATA_TYPE_HALF) {
                auto ptr = GetFloatFromRawBuffer(buf);
                CUDA_CHECK(cudaMemcpy(tempbufs_[0].ptr,
                                      ptr.get(),
                                      buf.GetDataCount() * DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT),
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(tempbufs_[1].ptr,
                                      buf.force_to<void*>(),
                                      buf.GetDataCount() * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF),
                                      cudaMemcpyHostToDevice));
            }
        }

        cublas_fp32_ = std::make_shared<cublasMMWrapper>(context_->GetCublasHandle(), context_->GetCublasLtHandle());
        cublas_fp16_ = std::make_shared<cublasMMWrapper>(context_->GetCublasHandle(), context_->GetCublasLtHandle());
        cublas_fp32_->setFP32GemmConfig();
        cublas_fp16_->setFP16GemmConfig();
    }

    return TNN_OK;
}

Status CudaMatMulLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status RunCudaGEMVKernel(const void* in_a_ptr, const void* in_b_ptr, void* out_ptr,
                         const int M, const int K, const int N,
                         const DimsVector in_a_dims, const DimsVector in_b_dims,
                         DataType dtype, CudaContext* context) {
    auto stream = context->GetStream();

    int size[3];
    int stride_a[3];
    int stride_b[3];

    int i = 0;
    for (; i < in_a_dims.size() - 2; i++) {
        size[i] = std::max(in_a_dims[i], in_b_dims[i]);
        stride_a[i] = in_a_dims[i] == 1 ? 0 : 1;
        stride_b[i] = in_b_dims[i] == 1 ? 0 : 1;
    }
    for (; i < 3; i++) {
        size[i] = 1;
        stride_a[i] = 0;
        stride_b[i] = 0;
    }

    if (stride_a[0] == 0 && stride_a[1] == 0 && stride_a[2] == 0) {
        stride_a[2] = 1;
    }
    if (stride_b[0] == 0 && stride_b[1] == 0 && stride_b[2] == 0) {
        stride_b[2] = 1;
    }

    dim3 dimGrid(K/BLOCK_DIM, N/BLOCK_DIM, size[0]*stride_a[0]+size[1]*stride_a[1]+size[2]*stride_a[2]);
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);

    int type_size = DataTypeUtils::GetBytesSize(dtype);
    int cur_workspace_size = (size[0]*stride_a[0]+size[1]*stride_a[1]+size[2]*stride_a[2]) * K * N * type_size;

    context->SetWorkspaceSize(cur_workspace_size);
    if (dtype == DataType::DATA_TYPE_FLOAT) {
        matmul_transpose_kernel<<<dimGrid, dimBlock, 0, stream>>>((float*)context->GetWorkspace(),
        (float*)in_a_ptr, K, N);
    } else {
        matmul_transpose_kernel<<<dimGrid, dimBlock, 0, stream>>>((__half*)context->GetWorkspace(),
        (__half*)in_a_ptr, K, N);
    }

    dim3 grid;
    grid.x = size[0] * size[1] * size[2];
    grid.y = (N + TNN_CUDA_NUM_THREADS - 1) / TNN_CUDA_NUM_THREADS;
    grid.z = (K + TNN_CUDA_NUM_THREADS - 1) / TNN_CUDA_NUM_THREADS;

    CUDA_CHECK(cudaMemsetAsync(out_ptr, 0, size[0] * size[1] * size[2] * N * type_size, stream));

    if (dtype == DataType::DATA_TYPE_FLOAT) {
        matmul_batched_gemv_kernel<<<grid, TNN_CUDA_NUM_THREADS, 0, stream>>>(
            (float*)context->GetWorkspace(), (float*)in_b_ptr, (float*)out_ptr, stride_a[0], stride_a[1],
            stride_a[2], stride_b[0], stride_b[1], stride_b[2], size[1], size[2], N, K);
    } else {
        matmul_batched_gemv_kernel_fp16<<<grid, TNN_CUDA_NUM_THREADS, 0, stream>>>(
            (__half*)context->GetWorkspace(), (float*)in_b_ptr, (__half*)out_ptr, stride_a[0], stride_a[1],
            stride_a[2], stride_b[0], stride_b[1], stride_b[2], size[1], size[2], N, K);
    }

    return TNN_OK;
}


Status CudaMatMulLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<MatMulLayerParam *>(param_);
    if (!param) {
        LOGE("Error: Unable to Get Param of CUDA MatMul Layer.\n");
        return Status(TNNERR_LAYER_ERR, "Error: Unable to Get Param of CUDA MatMul Layer.");
    }

    // Step 1: Prepare for CUDA MatMul.
    int M, K, N;
    int B = 1, B_in_a = 1, B_in_b = 1;   // Batch
    DimsVector in_a_dims, in_b_dims;
    DataType compute_dtype;

    void* in_a_ptr, *in_b_ptr;
    void* out_ptr = outputs[0]->GetHandle().base;

    if (inputs.size() == 1) {
        auto resource = dynamic_cast<MatMulLayerResource *>(resource_);
        if (!resource) {
            LOGE("Error: Unable to Get Resource of CUDA MatMul Layer.\n");
            return Status(TNNERR_LAYER_ERR, "Error: Unable to Get Resource of CUDA MatMul Layer.");
        }
        compute_dtype = inputs[0]->GetBlobDesc().data_type;
        if (param->weight_position == 0) {
            in_a_dims  = resource->weight.GetBufferDims();
            in_b_dims  = inputs[0]->GetBlobDesc().dims;
            for (int i=0; i<in_b_dims.size()-2; i++) {
                B_in_b *= in_b_dims[i];
            }
            in_a_ptr   = compute_dtype == DATA_TYPE_FLOAT ? tempbufs_[0].ptr : tempbufs_[1].ptr;
            in_b_ptr   = inputs[0]->GetHandle().base;
        } else {  //param->weight_position == 1
            in_a_dims  = inputs[0]->GetBlobDesc().dims;
            in_b_dims  = resource->weight.GetBufferDims();
            for (int i=0; i<in_a_dims.size()-2; i++) {
                B_in_a *= in_a_dims[i];
            }
            in_a_ptr   = inputs[0]->GetHandle().base;
            in_b_ptr   = compute_dtype == DATA_TYPE_FLOAT ? tempbufs_[0].ptr : tempbufs_[1].ptr;
        }
    } else {  //inputs.size() == 2
        if (inputs[0]->GetBlobDesc().data_type != inputs[1]->GetBlobDesc().data_type) {
            LOGE("Error: MatMul input Mat A and B has different data type, which is Not Supported by TNN cuda ACC.\n");
            return Status(TNNERR_MODEL_ERR, "Error: MatMul input Mat A and B has different data type, which is Not Supported by TNN cuda ACC.");
        }
        compute_dtype = inputs[0]->GetBlobDesc().data_type;
        in_a_dims     = inputs[0]->GetBlobDesc().dims;
        in_b_dims     = inputs[1]->GetBlobDesc().dims;
        for (int i=0; i<in_a_dims.size()-2; i++) {
            B_in_a   *= in_a_dims[i];
        }
        for (int i=0; i<in_b_dims.size()-2; i++) {
            B_in_b   *= in_b_dims[i];
        }
        in_a_ptr      = inputs[0]->GetHandle().base;
        in_b_ptr      = inputs[1]->GetHandle().base;
    }
    M = in_b_dims[in_b_dims.size() - 1];
    K = in_a_dims[in_a_dims.size() - 1];
    N = in_a_dims[in_a_dims.size() - 2];

    if (B_in_a != B_in_b) {
        if (B_in_b == 1) {
            // Treated As single batch GEMM
            N *= B_in_a;
            B = 1;
        } else {
            LOGE("Error: MatMul input Mat A and B has different multi-batch, which is Not Supported by TNN cuda ACC.\n");
            return Status(TNNERR_MODEL_ERR, "Error: MatMul input Mat A and B has different multi-batch, which is Not Supported by TNN cuda ACC.");
        }
    } else {
        // Batched-GEMM
        B = B_in_a;
    }
    if (compute_dtype != DataType::DATA_TYPE_FLOAT && compute_dtype != DataType::DATA_TYPE_HALF) {
        LOGE("Error: MatMul input Mat A and B has data type other than float and half, which is Not Supported by TNN cuda ACC.\n");
        return Status(TNNERR_MODEL_ERR, "Error: MatMul input Mat A and B has data type other than float and half, which is Not Supported by TNN cuda ACC.");
    }

    // MatMul with dynamic N of MNK:
    // Used in NLP models like BERT.
    //
    // In Bert Multi-Head Attention Module, there will be MatMuls with input like
    // [Batch*Max_Seq_len, 3*Hidden_Size] * [3*Hidden_Size, Hidden_Size]
    // Batch*Max_Seq_len is not fulfilled with Sequences of Length : Max_Seq_Len,
    // Thus, only Vaild Positions of Batch*Seq_len is actually meaningful.
    // We have implemented a Method to Move all Valid Sequences to the first N-current positions of Batch*Max_Seq_Len
    // We call this "Dense Mode"
    // Under Dense Mode, we will have an extra infomation called "bert_current_total_seqlen" stored in Context.
    // We get N-current from "bert_current_total_seqlen" from Context, and Implement MatMul only on the first N-current elements.
    if (B == 1) {
        auto& info_map = context_->GetExtraInfoMap();
        if (info_map.find("int_transformer_runtime_token_num") != info_map.end()) {
            auto rt_token_num = any_cast<int>(info_map["int_transformer_runtime_token_num"]);
            // Use dense mode only when rt_token_num is valid.
            if (rt_token_num > 0) {
                N = rt_token_num;
            }
        }
    }


    // Step 2: Run MatMul Kernels.
    if (B == 1) {
        if (M == 1 && in_a_dims.size() == in_b_dims.size()) {
            // Special GEMV case, Use hand-written CUDA kernels here.
            return RunCudaGEMVKernel(in_a_ptr, in_b_ptr, out_ptr, M, K, N, in_a_dims, in_b_dims, compute_dtype, context_);
        }

        // Standard GEMM Cases.
        if (compute_dtype == DATA_TYPE_FLOAT) {
            // Traditional CUBLAS Version
            //float alpha = 1.0;
            //float beta  = 0.0;
            //CUBLAS_CHECK(cublasSgemm(context_->GetCublasHandle(),
            //            CUBLAS_OP_N, CUBLAS_OP_N,
            //            M, N, K, &alpha, (float*)in_b_ptr, M, (float*)in_a_ptr, K,
            //            &beta, (float*)out_ptr, M));
            // New CUBLAS Wrapper Version
            cublas_fp32_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                               (float*)in_b_ptr, M, (float*)in_a_ptr, K, (float*)out_ptr, M, context_->GetStream());
        } else { // HALF
            // Traditional CUBLAS Version
            //__half alpha = __half(1.f);
            //__half beta  = __half(0.f);
            //CUBLAS_CHECK(cublasHgemm(context_->GetCublasHandle(),
            //             CUBLAS_OP_N, CUBLAS_OP_N,
            //             M, N, K, &alpha, (__half*)in_b_ptr, M, (__half*)in_a_ptr, K,
            //             &beta, (__half*)out_ptr, M));
            // New CUBLAS Wrapper Version
            cublas_fp16_->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                               (__half*)in_b_ptr, M, (__half*)in_a_ptr, K, (__half*)out_ptr, M, context_->GetStream());
        }
    } else {
        // B != 1, Batched-GEMM
        if (compute_dtype == DATA_TYPE_FLOAT) {
            float alpha = 1.0;
            float beta  = 0.0;
            CUBLAS_CHECK(cublasSgemmStridedBatched(context_->GetCublasHandle(),
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         M, N, K, &alpha, (float*)in_b_ptr, M, K*M, (float*)in_a_ptr, K, N*K,
                         &beta, (float*)out_ptr, M, N*M, B));
        } else { // HALF
            __half alpha = __half(1.f);
            __half beta  = __half(0.f);
            CUBLAS_CHECK(cublasHgemmStridedBatched(context_->GetCublasHandle(),
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         M, N, K, &alpha, (__half*)in_b_ptr, M, K*M, (__half*)in_a_ptr, K, N*K,
                         &beta, (__half*)out_ptr, M, N*M, B));
        }
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(MatMul, LAYER_MATMUL);

}  // namespace TNN_NS
