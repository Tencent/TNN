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

#include <cmath>
#include "tnn/device/cuda/acc/compute/compute.h"
#include "tnn/device/cuda/acc/compute/reduce_utils.cuh"

namespace TNN_NS {

__global__ void addBiasResidualPostLayerNormV2(float* out,
                                               const float* __restrict input_1,
                                               const float* __restrict input_2,
                                               const float* __restrict bias,
                                               const float* __restrict gamma,
                                               const float* __restrict beta,
                                               const float layernorm_eps,
                                               int         n)
{
    const int ite = 4;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out[ite];

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id   = i * blockDim.x + tid;
        int id       = bid * n + col_id;
        local_out[i] = (float)(input_1[id] + __ldg(&input_2[id]) + __ldg(&bias[col_id]));
        sum += local_out[i];
    }

    mean = blockReduceSum<float>(sum);
    if (tid == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < ite; i++) {
        float diff = local_out[i] - s_mean;
        var += diff * diff;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id     = bid * n + col_id;
        out[id] =
            (float)((local_out[i] - s_mean) * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id]));
    }
}

__global__ void addBiasResidualPostLayerNormV2(half* out,
                                               const half* __restrict input_1,
                                               const half* __restrict input_2,
                                               const half* __restrict bias,
                                               const half* __restrict gamma,
                                               const half* __restrict beta,
                                               const float layernorm_eps,
                                               int         n)
{
    using T2             = half2;
    const int        ite = 4;
    const int        tid = threadIdx.x;
    const int        bid = blockIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    T2               local_out_half2[ite];

    T2*       out_ptr    = (T2*)out;
    const T2* input1_ptr = (const T2*)input_1;
    const T2* input2_ptr = (const T2*)input_2;
    const T2* bias_ptr   = (const T2*)bias;
    const T2* gamma_ptr  = (const T2*)gamma;
    const T2* beta_ptr   = (const T2*)beta;

    // float sum = 0.0f;
    T2 sum = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id         = i * blockDim.x + tid;
        int id             = bid * n / 2 + col_id;
        local_out_half2[i] = input1_ptr[id] + __ldg(&input2_ptr[id]) + __ldg(&bias_ptr[col_id]);
        sum                = sum + local_out_half2[i];
    }

    mean = blockReduceSum<float>((float)(sum.x + sum.y));
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float var      = 0.0f;
    T2    s_mean_2 = __float2half2_rn(s_mean);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        local_out_half2[i] = local_out_half2[i] - s_mean_2;
        float v1           = (float)local_out_half2[i].x;
        float v2           = (float)local_out_half2[i].y;
        var += v1 * v1 + v2 * v2;
    }

    variance = blockReduceSum<float>(var);
    if (tid == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    T2 s_var_2 = __float2half2_rn(s_variance);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id  = i * blockDim.x + tid;
        int id      = bid * n / 2 + col_id;
        out_ptr[id] = local_out_half2[i] * s_var_2 * __ldg(&gamma_ptr[col_id]) + __ldg(&beta_ptr[col_id]);
    }
}

template<typename T, int N>
__global__ void addBiasResidualPostLayerNorm(
    T* out, const T* input_1, const T* input_2, const T* bias, const T* gamma, const T* beta, const float layernorm_eps, int m, int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;
    float            local_out_cache[N];

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = (float)(input_1[blockIdx.x * n + idx] + input_2[blockIdx.x * n + idx] + __ldg(&bias[idx]));
        mean += local_out;
        // save local_out to local_out_cache to save some recompute
        local_out_cache[i] = local_out;
        idx += blockDim.x;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        variance += (local_out - s_mean) * (local_out - s_mean);
        idx += blockDim.x;
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = variance / n + layernorm_eps;
    }
    __syncthreads();

#pragma unroll N
    for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
        float local_out = local_out_cache[i];
        out[blockIdx.x * n + idx] =
            (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[idx])) + (float)(__ldg(&beta[idx])));
        idx += blockDim.x;
    }
}

__global__ void generalAddBiasResidualPostLayerNorm(float*       out,
                                                    const float* input_1,
                                                    const float* input_2,
                                                    const float* bias,
                                                    const float* gamma,
                                                    const float* beta,
                                                    const float  layernorm_eps,
                                                    int          m,
                                                    int          n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = (float)(input_1[blockIdx.x * n + idx] + input_2[blockIdx.x * n + idx] + __ldg(&bias[idx]));
        mean += local_out;
        // save local_out to out to save some recompute
        out[blockIdx.x * n + idx] = local_out;
    }

    mean = blockReduceSum<float>(mean);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = out[blockIdx.x * n + idx];
        variance += (local_out - s_mean) * (local_out - s_mean);
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        float local_out = out[blockIdx.x * n + idx];
        out[blockIdx.x * n + idx] =
            (float)(((local_out - s_mean) * s_variance) * (float)(__ldg(&gamma[idx])) + (float)(__ldg(&beta[idx])));
    }
}

__global__ void generalAddBiasResidualPostLayerNorm(half*       out,
                                                    const half* input_1,
                                                    const half* input_2,
                                                    const half* bias,
                                                    const half* gamma,
                                                    const half* beta,
                                                    const float  layernorm_eps,
                                                    int          m,
                                                    int          n)
{
    using T2 = half2;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float            mean     = 0.0f;
    float            variance = 0.0f;

    T2*       out_ptr    = (T2*)out;
    const T2* input1_ptr = (const T2*)input_1;
    const T2* input2_ptr = (const T2*)input_2;
    const T2* bias_ptr   = (const T2*)bias;
    const T2* gamma_ptr  = (const T2*)gamma;
    const T2* beta_ptr   = (const T2*)beta;

    float local_out = 0.0f;
    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        T2     tmp           = input1_ptr[id] + input2_ptr[id] + __ldg(&bias_ptr[idx]);
        float2 local_out_fp2 = __half22float2(tmp);
        local_out += local_out_fp2.x;
        local_out += local_out_fp2.y;
        // save tmp to out_ptr to save some recomputation
        out_ptr[id] = tmp;
    }

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = __half22float2(out_ptr[id]);
        variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
        variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
    }

    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + layernorm_eps);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
        int    id            = blockIdx.x * n / 2 + idx;
        float2 local_out_fp2 = __half22float2(out_ptr[id]);
        float2 gamma_val     = __half22float2(__ldg(&gamma_ptr[idx]));
        float2 beta_val      = __half22float2(__ldg(&beta_ptr[idx]));
        local_out_fp2.x      = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
        local_out_fp2.y      = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
        out_ptr[id]          = __float22half2_rn(local_out_fp2);
    }
}

template<>
void invokeAddBiasResidualLayerNorm(float*       out,
                                    const float* input_1,
                                    const float* input_2,
                                    const float* bias,
                                    const float* gamma,
                                    const float* beta,
                                    const float  layernorm_eps,
                                    int          m,
                                    int          n,
                                    cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(std::min(n, 1024));
    if (n == 768 || n == 1024) {
        addBiasResidualPostLayerNormV2
            <<<grid, n / 4, 0, stream>>>(out, input_1, input_2, bias, gamma, beta, layernorm_eps, n);
    }
    else {
        block.x       = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            addBiasResidualPostLayerNorm<float, 1>
                <<<grid, block, 0, stream>>>(out, input_1, input_2, bias, gamma, beta, layernorm_eps, m, n);
        }
        else if (num_trips == 2) {
            addBiasResidualPostLayerNorm<float, 2>
                <<<grid, block, 0, stream>>>(out, input_1, input_2, bias, gamma, beta, layernorm_eps, m, n);
        }
        else {
            generalAddBiasResidualPostLayerNorm
                <<<grid, block, 0, stream>>>(out, input_1, input_2, bias, gamma, beta, layernorm_eps, m, n);
        }
    }
}

template <>
void invokeAddBiasResidualLayerNorm(half* out,
                                    const half*  input_1,
                                    const half*  input_2,
                                    const half*  bias,
                                    const half*  gamma,
                                    const half*  beta,
                                    const float  layernorm_eps,
                                    int          m,
                                    int          n,
                                    cudaStream_t stream) {
    dim3 grid(m);
    dim3 block(std::min(n, 1024));

    if (n == 768 || n == 1024) {
        addBiasResidualPostLayerNormV2
            <<<grid, n / 8, 0, stream>>>(out, input_1, input_2, bias, gamma, beta, layernorm_eps, n);
    }
    else {
        block.x       = std::min(n, 1024);
        int num_trips = (n + block.x - 1) / block.x;
        if (num_trips == 1) {
            addBiasResidualPostLayerNorm<half, 1>
                <<<grid, block, 0, stream>>>(out, input_1, input_2, bias, gamma, beta, layernorm_eps, m, n);
        }
        else if (num_trips == 2) {
            addBiasResidualPostLayerNorm<half, 2>
                <<<grid, block, 0, stream>>>(out, input_1, input_2, bias, gamma, beta, layernorm_eps, m, n);
        }
        else {
            generalAddBiasResidualPostLayerNorm
                <<<grid, block, 0, stream>>>(out, input_1, input_2, bias, gamma, beta, layernorm_eps, m, n);
        }
    }
}


__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__inline__ __device__ float tanh_opt(float x)
{
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
#else
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

template<typename T>
__inline__ __device__ T gelu(T x)
{
    float cdf = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (x + 0.044715f * x * x * x))));
    return x * cdf;
}

template<>
__inline__ __device__ half2 gelu(half2 val)
{
    half2  val_pow3 = __hmul2(val, __hmul2(val, val));
    float2 tmp_pow  = __half22float2(val_pow3);
    float2 tmp      = __half22float2(val);

    tmp.x = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
    tmp.y = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
    return __hmul2(val, __float22half2_rn(tmp));
}

template<typename T>
__global__ void addBiasGelu(T* out, const T* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T val = out[id];
        if (bias != nullptr) {
            T reg_bias = __ldg(&bias[id % n]);
            val        = val + reg_bias;
        }
        out[id] = (T)(gelu(val));
    }
}

template<>
__global__ void addBiasGelu(half* out, const half* __restrict bias, int m, int n)
{
    half2*       out_ptr  = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 val = out_ptr[id];
        if (bias != nullptr) {
            half2 reg_bias = __ldg(&bias_ptr[id % n]);
            val            = __hadd2(val, reg_bias);
        }
        out_ptr[id] = gelu(val);
    }
}

template<typename T>
void invokeAddBiasGelu(T* out, const T* bias, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    addBiasGelu<T><<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
}

template void invokeAddBiasGelu(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBiasGelu(half* out, const half* bias, const int m, const int n, cudaStream_t stream);


template<typename T2, int N>
__global__ void addBiasGeluV2(T2* out, const T2* __restrict bias, const int size)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < size; id += blockDim.x * gridDim.x) {
        T2 val = out[id];
        if (bias != nullptr) {
            T2 reg_bias = __ldg(&bias[id % N]);
            val         = __hadd2(val, reg_bias);
        }
        out[id] = gelu(val);
    }
}

template<typename T2, int N, int ELEMENT_PER_ROUND>
__global__ void addBiasGeluV3(T2* out, const T2* __restrict bias, const int size)
{
    T2 buffer[ELEMENT_PER_ROUND];
    T2 tmp_bias[ELEMENT_PER_ROUND];
    for (int id = blockIdx.x * blockDim.x * ELEMENT_PER_ROUND + threadIdx.x * ELEMENT_PER_ROUND; id < size;
         id += blockDim.x * gridDim.x * ELEMENT_PER_ROUND) {
#pragma unroll
        for (int i = 0; i < ELEMENT_PER_ROUND; i++) {
            buffer[i] = out[id + i];
            if (bias != nullptr) {
                tmp_bias[i] = __ldg(&bias[(id + i) % N]);
            }
        }
#pragma unroll
        for (int i = 0; i < ELEMENT_PER_ROUND; i++) {
            if (bias != nullptr) {
                buffer[i] = __hadd2(buffer[i], tmp_bias[i]);
            }
            out[id + i] = gelu(buffer[i]);
        }
    }
}

#define ADD_BIAS_GELU(HALF_N, ELEMENT_PER_ROUND)                                                                       \
    case HALF_N:                                                                                                       \
        if (ELEMENT_PER_ROUND > 1) {                                                                                   \
            grid.x = grid.x / ELEMENT_PER_ROUND;                                                                       \
            addBiasGeluV3<T2, HALF_N, ELEMENT_PER_ROUND>                                                               \
                <<<grid, block, 0, stream>>>((T2*)out, (const T2*)bias, m * half_n);                                   \
        }                                                                                                              \
        else {                                                                                                         \
            addBiasGeluV2<T2, HALF_N><<<grid, block, 0, stream>>>((T2*)out, (const T2*)bias, m * half_n);              \
        }                                                                                                              \
        break;

template<typename T>
void invokeAddBiasGeluV2(T* out, const T* bias, const int m, const int n, cudaStream_t stream)
{
    if (n % 2 == 0 && sizeof(T) == 2) {
        const int half_n = n / 2;
        dim3      block, grid;
        block.x  = std::min(half_n, 512);
        grid.x   = (m * half_n + (block.x - 1)) / block.x;
        using T2 = half2;

        if (grid.x >= 512) {
            switch (half_n) {
                ADD_BIAS_GELU(256, 1)
                ADD_BIAS_GELU(512, 1)
                ADD_BIAS_GELU(1024, 1)
                ADD_BIAS_GELU(1536, 1)
                ADD_BIAS_GELU(2048, 1)
                ADD_BIAS_GELU(4096, 2)
                ADD_BIAS_GELU(8192, 2)
                ADD_BIAS_GELU(16384, 2)
                ADD_BIAS_GELU(24576, 2)
                ADD_BIAS_GELU(40960, 4)
                default:
                    invokeAddBiasGelu(out, bias, m, n, stream);
                    break;
            }
        }
        else {
            switch (half_n) {
                ADD_BIAS_GELU(256, 1)
                ADD_BIAS_GELU(512, 1)
                ADD_BIAS_GELU(1024, 1)
                ADD_BIAS_GELU(1536, 1)
                ADD_BIAS_GELU(2048, 1)
                ADD_BIAS_GELU(4096, 1)
                ADD_BIAS_GELU(8192, 2)
                ADD_BIAS_GELU(16384, 2)
                ADD_BIAS_GELU(24576, 2)
                ADD_BIAS_GELU(40960, 2)
                default:
                    invokeAddBiasGelu(out, bias, m, n, stream);
                    break;
            }
        }
    }
    else {
        invokeAddBiasGelu(out, bias, m, n, stream);
    }
}

#undef ADD_BIAS_GELU

template void invokeAddBiasGeluV2(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBiasGeluV2(half* out, const half* bias, const int m, const int n, cudaStream_t stream);



template<typename T>
FfnLayer<T>::FfnLayer(cublasMMWrapper* cublas_wrapper) :
    cublas_wrapper_in_(std::make_shared<cublasMMWrapper>(*cublas_wrapper)),
    cublas_wrapper_out_(std::make_shared<cublasMMWrapper>(*cublas_wrapper)) {
}

template<typename T>
void FfnLayer<T>::forward(T* output,
                          T* input,
                          T* ffn_matmul_in,
                          T* ffn_bias,
                          T* ffn_matmul_out,
                          T* inter_buf,
                          int token_num,
                          int hidden_dimension,
                          int inter_size,
                          cudaStream_t stream) {
    // input tensors:
    //      ffn_input [token_num, hidden_dimension],

    // output tensors:
    //      ffn_output [token_num, hidden_dimension],

    cublas_wrapper_in_->Gemm(CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             inter_size,
                             token_num,
                             hidden_dimension,
                             ffn_matmul_in,
                             inter_size,
                             input,
                             hidden_dimension,
                             inter_buf,
                             inter_size,
                             stream);

    invokeAddBiasActivation(token_num, inter_size, inter_buf, ffn_bias, stream);

    cublas_wrapper_out_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              hidden_dimension,
                              token_num,
                              inter_size,
                              ffn_matmul_out,
                              hidden_dimension,
                              inter_buf,
                              inter_size,
                              output,
                              hidden_dimension,
                              stream);

}

template<typename T>
GeluFfnLayer<T>::GeluFfnLayer(cublasMMWrapper* cublas_wrapper) :
    FfnLayer<T>(cublas_wrapper) {
}

template<typename T>
void GeluFfnLayer<T>::invokeAddBiasActivation(const int token_num, const int inter_size, T* inter_buf, const T* bias, cudaStream_t stream) {
    invokeAddBiasGeluV2<T>(inter_buf, bias, token_num, inter_size, stream);
}

template class GeluFfnLayer<float>;
template class GeluFfnLayer<half>;

__global__ void trt_add_QKV_bias(half2*       qkv_buf,
                                 const half2* Q,
                                 const half2* bias_Q,
                                 const half2* K,
                                 const half2* bias_K,
                                 const half2* V,
                                 const half2* bias_V,
                                 const int    valid_word_num,
                                 const int    head_num,
                                 const int    size_per_head)
{
    // Add bias, and then transpose from
    // [3, valid_word_num, head, size] -> [valid_word_num, head, 3, size]

    // const int seq_id = blockIdx.x % valid_word_num;
    // const int qkv_id = (blockIdx.x - seq_id) / valid_word_num;
    const int seq_id = blockIdx.x;

    for (int index = threadIdx.x; index < head_num * size_per_head; index += blockDim.x) {
        const int size_id = index % size_per_head;
        const int head_id = (index - size_id) / size_per_head;

        const int target_offset = blockIdx.x * head_num * 3 * size_per_head + head_id * 3 * size_per_head;
        const int src_id        = seq_id * head_num * size_per_head + index;

        qkv_buf[target_offset + 0 * size_per_head + size_id] = Q[src_id] + bias_Q[index];
        qkv_buf[target_offset + 1 * size_per_head + size_id] = K[src_id] + bias_K[index];
        qkv_buf[target_offset + 2 * size_per_head + size_id] = V[src_id] + bias_V[index];
    }
}

template<typename T>
void invokeTrtAddQkvBias(size_t token_num, int head_num, int size_per_head,
                         T* qkv_buf, T* q_buf, T* k_buf, T* v_buf,
                         T* q_bias, T* k_bias, T* v_bias, cudaStream_t stream) {
    dim3 grid(token_num);
    dim3 block(min((int)(head_num * size_per_head / 2), 512));

    trt_add_QKV_bias<<<grid, block, 0, stream>>>((half2*)qkv_buf,
                                                 (const half2*)q_buf,
                                                 (const half2*)q_bias,
                                                 (const half2*)k_buf,
                                                 (const half2*)k_bias,
                                                 (const half2*)v_buf,
                                                 (const half2*)v_bias,
                                                 token_num,
                                                 head_num,
                                                 size_per_head / 2);
}

__global__ void getTrtPaddingOffsetKernel(int*       trt_mha_padding_offset,
                                          const int* sequence_length,
                                          const int  request_batch_size,
                                          const int  request_seq_len)
{
    // use for get tensorrt fused mha padding offset
    // when we keep the padding

    extern __shared__ int tmp_offset[];
    if (threadIdx.x == 0) {
        tmp_offset[0] = 0;
        for (int i = 0; i < request_batch_size; i++) {
            tmp_offset[i * 2 + 1] = tmp_offset[i * 2] + sequence_length[i];
            tmp_offset[i * 2 + 2] = request_seq_len * (i + 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 2 * request_batch_size + 1; i += blockDim.x) {
        trt_mha_padding_offset[i] = tmp_offset[i];
    }
}

void invokeGetTrtPaddingOffset(int*         trt_mha_padding_offset,
                               const int*   sequence_length,
                               const int    request_batch_size,
                               const int    request_seq_len,
                               cudaStream_t stream)
{
    getTrtPaddingOffsetKernel<<<1, 256, sizeof(int) * (2 * request_batch_size + 1), stream>>>(
        trt_mha_padding_offset, sequence_length, request_batch_size, request_seq_len);
}

template<typename T>
__global__ void getTrtPaddingOffsetFromMaskKernel(int*       trt_mha_padding_offset,
                                                  const T*   mask,
                                                  const int  ld_mask,
                                                  const int  request_batch_size,
                                                  const int  request_seq_len)
{
    // use for get tensorrt fused mha padding offset
    // when we keep the padding

    extern __shared__ int sequence_length[];
    for (int i = threadIdx.x; i < request_batch_size; i += blockDim.x) {
        if (mask != nullptr) {
            const T* b_mask = mask + i * ld_mask;
            int len = 0;
            for (int j = 0; j < request_seq_len; ++j) {
                if ((float(b_mask[j]) - 0.0) > -1e-5 && (float(b_mask[j]) - 0.0) < 1e-5) {
                    ++len;
                } else {
                    break;
                }
            }
            sequence_length[i] = len;
        } else {
            sequence_length[i] = request_seq_len;
        }
    }
    __syncthreads();

    int *tmp_offset = sequence_length + request_batch_size;
    if (threadIdx.x == 0) {
        tmp_offset[0] = 0;
        for (int i = 0; i < request_batch_size; i++) {
            tmp_offset[i * 2 + 1] = tmp_offset[i * 2] + sequence_length[i];
            tmp_offset[i * 2 + 2] = request_seq_len * (i + 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 2 * request_batch_size + 1; i += blockDim.x) {
        trt_mha_padding_offset[i] = tmp_offset[i];
    }
}

template<typename T>
void invokeGetTrtPaddingOffsetFromMask(int*         trt_mha_padding_offset,
                                       const T*     mask,
                                       const int    ld_mask,
                                       const int    request_batch_size,
                                       const int    request_seq_len,
                                       cudaStream_t stream)
{
    getTrtPaddingOffsetFromMaskKernel<<<1, 256, sizeof(int) * (3 * request_batch_size + 1), stream>>>(
        trt_mha_padding_offset, mask, ld_mask, request_batch_size, request_seq_len);
}

#if 0  // Fused Attention has 100mb + volume
template<typename T>
FusedAttentionLayer<T>::FusedAttentionLayer(size_t           head_num,
                                            size_t           size_per_head,
                                            size_t           d_model,
                                            float            q_scaling,
                                            int              sm,
                                            cublasMMWrapper* cublas_wrapper) :
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    q_scaling_(q_scaling),
    sm_(sm),
    cublas_wrapper_(std::make_shared<cublasMMWrapper>(*cublas_wrapper)) {
    if ((sm_ == 70 || sm_ == 86 || sm_ == 80 || sm_ == 75 || sm_ == 72) && size_per_head_ == 64) {
        dispatcher_fp16.reset(new FusedMHARunnerFP16v2(head_num_, size_per_head_, sm_, q_scaling_));
    } else {
        throw std::runtime_error(std::string("FusedAttentionLayer not support.\n"));
    }
    hidden_units_ = head_num_ * size_per_head_;
}

template<typename T>
void FusedAttentionLayer<T>::forward(T* attention_out,
                                     T* from_tensor,
                                     T* attention_mask,
                                     int* padding_offset,
                                     T* inter_buf,
                                     T* q_weight,
                                     T* k_weight,
                                     T* v_weight,
                                     T* o_weight,
                                     T* q_bias,
                                     T* k_bias,
                                     T* v_bias,
                                     int h_token_num,
                                     int max_seq_len,
                                     int batch_size,
                                     int ld_mask,
                                     cudaStream_t stream) {
    // input_tensors: [input_query (h_token_num, d_model),
    //                 attention_mask (batch, 1, seqlen, seqlen) or (batch, 1, seqlen),
    //                 padding_offset (batch + 1 or batch * 2 + 1))]

    const size_t m = h_token_num;
    int          k = d_model_;
    int          n = hidden_units_;

    T* q_buf_     = inter_buf;
    T* k_buf_     = q_buf_ + m*n;
    T* v_buf_     = k_buf_ + m*n;
    T* qkv_buf_   = v_buf_ + m*n;
    T* qkv_buf_2_ = qkv_buf_ + m*n * 3;
    int* padding_offset_ = reinterpret_cast<int*>(qkv_buf_2_ + m*n);

    // TODO: support batched gemm
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          q_weight,
                          n,
                          from_tensor,
                          k,
                          q_buf_,
                          n,
                          stream);
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          k_weight,
                          n,
                          from_tensor,
                          k,
                          k_buf_,
                          n,
                          stream);
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          v_weight,
                          n,
                          from_tensor,
                          k,
                          v_buf_,
                          n,
                          stream);

    invokeTrtAddQkvBias(m, head_num_, size_per_head_, qkv_buf_, q_buf_, k_buf_, v_buf_, q_bias, k_bias, v_bias, stream);

    int S = dispatcher_fp16->getSFromMaxSeqLen(max_seq_len);
    if(!dispatcher_fp16->isValid(S)) {
        throw std::runtime_error(std::string("FusedAttentionLayer max_seq_len not valid.\n"));
    }
    int B = batch_size;
    if (padding_offset == nullptr) {
        invokeGetTrtPaddingOffsetFromMask(padding_offset_, attention_mask, ld_mask, batch_size, max_seq_len, stream);

        // int seq_lenghts[] = {16,16};
        // int* d_seq_lengths;
        // cudaMalloc(&d_seq_lengths, batch_size*sizeof(int));
        // cudaStreamSynchronize(stream);
        // cudaMemcpy(d_seq_lengths, seq_lenghts, batch_size*sizeof(int), cudaMemcpyHostToDevice);
        // invokeGetTrtPaddingOffset(padding_offset_, d_seq_lengths, batch_size, max_seq_len, stream);
        // cudaFree(d_seq_lengths);

        // cudaStreamSynchronize(stream);
        // int *h_padding_offset = new int[batch_size*2+1];
        // cudaMemcpy(h_padding_offset, npadding_offset_, (batch_size*2+1)*sizeof(int), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < batch_size*2+1; ++i) {
        //     printf("[%d] %d\n", i, h_padding_offset[i]);
        // }
        // delete[] h_padding_offset;

        padding_offset = padding_offset_;
        B = batch_size * 2;
    }
    dispatcher_fp16->setup(S, B);
    dispatcher_fp16->run(qkv_buf_,
                         nullptr,
                         padding_offset,
                         nullptr,
                         qkv_buf_2_,
                         stream);

    k = hidden_units_;
    n = d_model_;
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          o_weight,
                          n,
                          qkv_buf_2_,
                          k,
                          attention_out,
                          n,
                          stream);
}

template class FusedAttentionLayer<half>;




// UnfusedAttentionLayer
template<typename T>
UnfusedAttentionLayer<T>::UnfusedAttentionLayer(size_t           head_num,
                                                size_t           size_per_head,
                                                size_t           d_model,
                                                float            q_scaling,
                                                int              sm,
                                                cublasMMWrapper* cublas_wrapper) :
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    q_scaling_(q_scaling),
    sm_(sm),
    cublas_wrapper_(std::make_shared<cublasMMWrapper>(*cublas_wrapper)) {
    hidden_units_ = head_num_ * size_per_head_;
}

template<typename T>
void UnfusedAttentionLayer<T>::forward(T* attention_out,
                                       T* from_tensor,
                                       T* attention_mask,
                                       int* padding_offset,
                                       T* inter_buf,
                                       T* q_weight,
                                       T* k_weight,
                                       T* v_weight,
                                       T* o_weight,
                                       T* q_bias,
                                       T* k_bias,
                                       T* v_bias,
                                       int h_token_num,
                                       int max_seq_len,
                                       int batch_size,
                                       int ld_mask,
                                       cudaStream_t stream) {
    const size_t m = h_token_num;
    int          k = d_model_;
    int          n = hidden_units_;
    bool mask_2d = false;
    if (ld_mask == max_seq_len * max_seq_len) {
        mask_2d = true;
    } else if (ld_mask != max_seq_len) {
        LOGE("Shape of Attention Mask should be [batch, 1, seq_len, seq_len] or [batch, 1, seq_len].\n");
        return;
    }

    // TOTAL: real_time_total_seq_len * hidden_units * 9 + batch*head_num*max_seq_len*max_seq_len;
    T* q_buf_     = inter_buf;
    T* k_buf_     = q_buf_ + m*n;
    T* v_buf_     = k_buf_ + m*n;
    T* q_buf_2_   = v_buf_ + m*n;
    T* k_buf_2_   = q_buf_2_ + batch_size * max_seq_len * hidden_units_;
    T* v_buf_2_   = k_buf_2_ + batch_size * max_seq_len * hidden_units_;
    T* qk_buf_    = v_buf_2_ + batch_size * max_seq_len * hidden_units_;
    T* qkv_buf_   = qk_buf_ + batch_size*head_num_*max_seq_len*max_seq_len;
    T* qkv_buf_2_ = qkv_buf_ + batch_size * max_seq_len * hidden_units_;

    // TODO: support batched gemm
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          q_weight,
                          n,
                          from_tensor,
                          k,
                          q_buf_,
                          n,
                          stream);
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          k_weight,
                          n,
                          from_tensor,
                          k,
                          k_buf_,
                          n,
                          stream);
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          v_weight,
                          n,
                          from_tensor,
                          k,
                          v_buf_,
                          n,
                          stream);

    if (padding_offset == nullptr) {
        // Sparse Mode
        invokeAddQKVBiasTranspose(q_buf_2_,
                                  k_buf_2_,
                                  v_buf_2_,
                                  q_buf_,
                                  q_bias,
                                  k_buf_,
                                  k_bias,
                                  v_buf_,
                                  v_bias,
                                  batch_size,
                                  max_seq_len,
                                  head_num_,
                                  size_per_head_,
                                  stream);
    } else {
        // Dense Mode
        cudaMemsetAsync(q_buf_2_, 0, 3 * batch_size * max_seq_len * hidden_units_ * sizeof(T), stream);
        invokeAddQKVBiasRebuildPadding(q_buf_,
                                       q_bias,
                                       k_buf_,
                                       k_bias,
                                       v_buf_,
                                       v_bias,
                                       q_buf_2_,
                                       k_buf_2_,
                                       v_buf_2_,
                                       batch_size,
                                       max_seq_len,
                                       head_num_,
                                       size_per_head_,
                                       m,
                                       padding_offset,
                                       stream);
    }

    float scalar = 1 / (std::sqrt(size_per_head_ * 1.0f) * q_scaling_);
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        batch_size * head_num_,
                                        max_seq_len,
                                        max_seq_len,
                                        size_per_head_,
                                        k_buf_2_,
                                        size_per_head_,
                                        max_seq_len * size_per_head_,
                                        q_buf_2_,
                                        size_per_head_,
                                        max_seq_len * size_per_head_,
                                        qk_buf_,
                                        max_seq_len,
                                        max_seq_len * max_seq_len,
                                        stream,
                                        scalar);

    // TODO (Relative Position Bias Not Supported)
    //if (use_relative_position_bias) {
    //    invokeAddRelativeAttentionBias(
    //        qk_buf_, relative_attention_bias, request_batch_size, head_num_, request_seq_len, stream_);
    //}

    invokeMaskedSoftMax(qk_buf_,
                        qk_buf_,
                        attention_mask,
                        batch_size,
                        max_seq_len,
                        max_seq_len,
                        head_num_,
                        (T)1.0f,
                        mask_2d,
                        stream);

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        batch_size * head_num_,
                                        size_per_head_,
                                        max_seq_len,
                                        max_seq_len,
                                        v_buf_2_,
                                        size_per_head_,
                                        max_seq_len * size_per_head_,
                                        qk_buf_,
                                        max_seq_len,
                                        max_seq_len * max_seq_len,
                                        qkv_buf_,
                                        size_per_head_,
                                        max_seq_len * size_per_head_,
                                        stream);

    if (padding_offset == nullptr) {
        // Sparse Mode
        invokeTransposeQKV(qkv_buf_2_, qkv_buf_, batch_size, max_seq_len, head_num_, size_per_head_, stream);
    } else {
        // Dense Mode
        invokeTransposeAttentionOutRemovePadding(qkv_buf_,
                                                 qkv_buf_2_,
                                                 m,
                                                 batch_size,
                                                 max_seq_len,
                                                 head_num_,
                                                 size_per_head_,
                                                 padding_offset,
                                                 stream);
    }

    // Output MatMul
    k = hidden_units_;
    n = d_model_;
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          n,
                          m,
                          k,
                          o_weight,
                          n,
                          qkv_buf_2_,
                          k,
                          attention_out,
                          n,
                          stream);
}

template class UnfusedAttentionLayer<half>;
template class UnfusedAttentionLayer<float>;

template<typename T>
void FlashAttentionLayer<T>::forward(T* devQKV,
                         T* output,
                         int32_t batch_size,
                         int32_t head_num, 
                         int32_t size_per_head, 
                         int32_t seq_len,
                         cudaStream_t stream) {
    if (batch_size != mOptBatchSize || seq_len != mOptSeqLen)
    {
        initializeSeqlens(batch_size, seq_len, mCuSeqLen.get(), stream);
    }
    size_t const total = mOptBatchSize * mOptSeqLen;
    runFMHFAKernel(devQKV, mCuSeqLen.get(), output, total, mSM, mKernels,
            mOptBatchSize, head_num, size_per_head, mOptSeqLen, stream);
}

template<typename T>
void FlashAttentionLayer<T>::createMHARunner()
{
    mKernels = getFMHAFlashCubinKernels(MHA_DATA_TYPE_FP16, mSM); 
}


template<typename T>
void FlashAttentionLayer<T>::allocateSeqlens(int32_t maxBatchSize)
{
    // allocate seqlens buffer
    if (!mCuSeqLen && maxBatchSize)
    {
        void* cudaMem{nullptr};
        cudaMalloc(&cudaMem, sizeof(int32_t) * (maxBatchSize + 1));
        make_cuda_shared(mCuSeqLen, cudaMem);
    }

    mMaxBatchSize = maxBatchSize;
}

template<typename T>
void FlashAttentionLayer<T>::initializeSeqlens(int32_t b, int32_t s, void* cu_seqlens_d, cudaStream_t stream)
{
    if (!b || !s)
    {
        return;
    }

    std::vector<int32_t> cuSeqLens(b + 1, 0);
    // Compute the prefix sum of the seqlen
    for (int32_t it = 0; it < b; it++)
    {
        cuSeqLens[it + 1] = cuSeqLens[it] + s;
    }

    cudaMemcpyAsync(
        cu_seqlens_d, cuSeqLens.data(), sizeof(int32_t) * cuSeqLens.size(), cudaMemcpyHostToDevice, stream);
    mOptBatchSize = b;
    mOptSeqLen = s;
}

template class FlashAttentionLayer<half>;

template<typename T>
void CrossAttentionLayer<T>::forward(T* devQ,
                         T* devKV,
                         T* output,
                         int32_t batch_size,
                         int32_t head_num, 
                         int32_t size_per_head, 
                         int32_t seq_len_q,
                         int32_t seq_len_kv,
                         cudaStream_t stream) {
    constexpr int32_t seqLenKvPadded = 128;
    if (batch_size != mOptBatchSize) {
        allocateSeqlens(batch_size);
    }
    if (batch_size != mOptBatchSize || seq_len_q != mOptSeqLenQ ||seq_len_kv != mOptSeqLenKV)
    {
        mOptSeqLenQ = seq_len_q;
        mOptSeqLenKV = seq_len_kv;
        initializeSeqlens(batch_size, seq_len_q, mCuSeqLenQ.get(), stream);
        initializeSeqlens(batch_size, seq_len_kv, mCuSeqLenKV.get(), stream);
    }
    runFMHCAKernel(devQ, devKV, mCuSeqLenQ.get(), mCuSeqLenKV.get(), output, mSM, mKernels,
            mOptBatchSize, head_num, size_per_head, mOptSeqLenQ, seqLenKvPadded, stream);
}

template<typename T>
void CrossAttentionLayer<T>::createMHARunner()
{
    mKernels = getFMHCACubinKernels(MHA_DATA_TYPE_FP16, mSM); 
}


template<typename T>
void CrossAttentionLayer<T>::allocateSeqlens(int32_t maxBatchSize)
{
    // allocate seqlens buffer
    if ((!mCuSeqLenQ || !mCuSeqLenKV) && maxBatchSize)
    {
        void* cudaMemQ{nullptr};
        void* cudaMemKV{nullptr};
        cudaMalloc(&cudaMemQ, sizeof(int32_t) * (maxBatchSize + 1));
        cudaMalloc(&cudaMemKV, sizeof(int32_t) * (maxBatchSize + 1));
        make_cuda_shared(mCuSeqLenQ, cudaMemQ);
        make_cuda_shared(mCuSeqLenKV, cudaMemKV);
    }

    mMaxBatchSize = maxBatchSize;
}

template<typename T>
void CrossAttentionLayer<T>::initializeSeqlens(int32_t b, int32_t s, void* cu_seqlens_d, cudaStream_t stream)
{
    if (!b || !s)
    {
        return;
    }

    std::vector<int32_t> cuSeqLens(b + 1, 0);
    // Compute the prefix sum of the seqlen
    for (int32_t it = 0; it < b; it++)
    {
        cuSeqLens[it + 1] = cuSeqLens[it] + s;
    }

    cudaMemcpyAsync(
        cu_seqlens_d, cuSeqLens.data(), sizeof(int32_t) * cuSeqLens.size(), cudaMemcpyHostToDevice, stream);
    mOptBatchSize = b;
}

template class CrossAttentionLayer<half>;
#endif  // end #if 0

}  // namespace TNN_NS
