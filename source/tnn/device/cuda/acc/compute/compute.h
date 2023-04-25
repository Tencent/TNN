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

#ifndef TNN_CUDA_COMPUTE_H_
#define TNN_CUDA_COMPUTE_H_

#include <float.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <memory>

#include <cuda_fp16.h>

#include "tnn/device/cuda/utils.cuh"
//#include "tnn/device/cuda/acc/compute/trt_fused_multihead_attention/qkvToContext.h"
//#include "tnn/device/cuda/acc/compute/trt_multihead_flash_attention/fmha_flash_attention/include/fmha_flash_attention.h"
//#include "tnn/device/cuda/acc/compute/trt_multihead_flash_attention/fmha.h"
#include "tnn/device/cuda/acc/compute/trt_unfused_multihead_attention/unfused_multihead_attention.h"

namespace TNN_NS {

template<typename T>
void invokeAddBiasResidualLayerNorm(T*           out,
                                    const T*     input_1,
                                    const T*     input_2,
                                    const T*     bias,
                                    const T*     gamma,
                                    const T*     beta,
                                    const float  layernorm_eps,
                                    const int    m,
                                    const int    n,
                                    cudaStream_t stream);

template<typename T>
void invokeAddBiasGelu(T* out, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeAddBiasGeluV2(T* out, const T* bias, const int m, const int n, cudaStream_t stream);



template<typename T>
class FfnLayer {
public:
    FfnLayer(cublasMMWrapper* cublas_wrapper);

    virtual void forward(T* output,
                         T* input,
                         T* ffn_matmul_in,
                         T* ffn_bias,
                         T* ffn_matmul_out,
                         T* inter_buf,
                         int token_num,
                         int hidden_dimension,
                         int inter_size,
                         cudaStream_t stream);

protected:
    virtual void invokeAddBiasActivation(const int token_num, const int inter_size, T* inter_buf, const T* bias, cudaStream_t stream) = 0;

    std::shared_ptr<cublasMMWrapper> cublas_wrapper_in_;
    std::shared_ptr<cublasMMWrapper> cublas_wrapper_out_;
};

template<typename T>
class GeluFfnLayer: public FfnLayer<T> {
public:
    GeluFfnLayer(cublasMMWrapper* cublas_wrapper);

private:
    void invokeAddBiasActivation(const int token_num, const int inter_size, T* inter_buf, const T* bias, cudaStream_t stream) override;
};

template<typename T>
class BaseAttentionLayer {
public:
    virtual void forward(T* attention_out,
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
                         cudaStream_t stream) = 0;
    virtual ~BaseAttentionLayer() = default;
}; // Class BaseAttentionLayer

#if 0  // Fused Attention has 100mb+ volume
template<typename T>
class FusedAttentionLayer : public BaseAttentionLayer<T> {
public:
    FusedAttentionLayer(size_t           head_num,
                        size_t           size_per_head,
                        size_t           d_model,
                        float            q_scaling,
                        int              sm,
                        cublasMMWrapper* cublas_wrapper);

    void forward(T* attention_out,
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
                 cudaStream_t stream) override;

private:
    size_t head_num_;
    size_t size_per_head_;
    size_t d_model_;
    float  q_scaling_;
    int    sm_;
    std::shared_ptr<cublasMMWrapper> cublas_wrapper_;

    size_t hidden_units_;
    std::unique_ptr<MHARunner> dispatcher_fp16;
};


template<typename T>
class UnfusedAttentionLayer : public BaseAttentionLayer<T> {
public:
    UnfusedAttentionLayer(size_t           head_num,
                          size_t           size_per_head,
                          size_t           d_model,
                          float            q_scaling,
                          int              sm,
                          cublasMMWrapper* cublas_wrapper);

    void forward(T* attention_out,
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
                 cudaStream_t stream) override;

private:
    size_t head_num_;
    size_t size_per_head_;
    size_t d_model_;
    float  q_scaling_;
    int    sm_;
    std::shared_ptr<cublasMMWrapper> cublas_wrapper_;

    size_t hidden_units_;
};

class FusedMultiHeadFlashAttentionKernel;

template <typename T>
struct CudaDeleter
{
    void operator()(T* buf)
    {
         cudaFree(buf);
    }
};

template <typename T>
using cuda_shared_ptr = std::shared_ptr<T>;

template <typename T>
void make_cuda_shared(cuda_shared_ptr<T>& ptr, void* cudaMem)
{
    ptr.reset(static_cast<T*>(cudaMem), CudaDeleter<T>());
}


template<typename T>
class FlashAttentionLayer {
public:
    FlashAttentionLayer(int sm) {
        mSM = sm;
        mOptBatchSize = 0;
        mOptSeqLen = 0;
        mMaxBatchSize = 0;
        mCuSeqLen = nullptr;
    }
    void forward(T* devQKV,
                 T* output,
                 int32_t batch_size,
                 int32_t head_num, 
                 int32_t size_per_head, 
                 int32_t seq_len,
                 cudaStream_t stream);
    void createMHARunner();
    void allocateSeqlens(int32_t maxBatchSize);
    void initializeSeqlens(int32_t b, int32_t s, void* cu_seqlens_d, cudaStream_t stream = 0);
    ~FlashAttentionLayer(){};
private:
    int32_t mOptBatchSize;
    int32_t mOptSeqLen;
    int32_t mMaxBatchSize;
    DataType mDataType; //{DataType::kFLOAT};
    int mSM;
    cuda_shared_ptr<void> mCuSeqLen;
    FusedMultiHeadFlashAttentionKernel const* mKernels;

    std::string const mLayerName;
    std::string mNamespace;
}; // Class FlashAttentionLayer
#endif  // end of #if 0

}   // namespace TNN_NS

#endif  // TNN_CUDA_COMPUTE_H_
