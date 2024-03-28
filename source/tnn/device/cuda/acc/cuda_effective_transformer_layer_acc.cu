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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(EffectiveTransformer, LAYER_EFFECTIVE_TRANSFORMER);

Status CudaEffectiveTransformerLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaEffectiveTransformerLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

__global__ void getPaddingOffsetKernel(int*    valid_word_num,
                                       int*       tmp_mask_offset,
                                       const int* sequence_length,
                                       const int  batch_size,
                                       const int  max_seq_len)
{
    // do cumulated sum
    int total_seq_len = 0;
    int cum_offset    = 0;
    int index         = 0;
    for (int i = 0; i < batch_size; i++) {
        const int seq_len = sequence_length[i];
        for (int j = 0; j < seq_len; j++) {
            tmp_mask_offset[index] = cum_offset;
            index++;
        }
        cum_offset += max_seq_len - seq_len;
        total_seq_len += seq_len;
    }
    valid_word_num[0] = total_seq_len;
}

void invokeGetPaddingOffset(int*      h_token_num,
                            int*      d_token_num,
                            int*         tmp_mask_offset,
                            const int*   sequence_lengths,
                            const int    batch_size,
                            const int    max_seq_len,
                            cudaStream_t stream)
{
    getPaddingOffsetKernel<<<1, 1, 0, stream>>>(
        d_token_num, tmp_mask_offset, sequence_lengths, batch_size, max_seq_len);
    CUDA_CHECK(cudaMemcpyAsync(h_token_num, d_token_num, sizeof(int), cudaMemcpyDeviceToHost, stream));
}

template<typename T>
__global__ void getPaddingOffsetFromMaskKernel(int*       valid_word_num,
                                               int*       tmp_mask_offset,
                                               const T*   mask,
                                               const int  batch_size,
                                               const int  max_seq_len,
                                               const int  ld_mask)
{
    extern __shared__ int sequence_length[];
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        const T* b_mask = mask + i * ld_mask;
        int len = 0;
        for (int j = 0; j < max_seq_len; ++j) {
            if ((float(b_mask[j]) - 0.0) > -1e-5 && (float(b_mask[j]) - 0.0) < 1e-5) {
                ++len;
            } else {
                break;
            }
        }
        sequence_length[i] = len;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // do cumulated sum
        int total_seq_len = 0;
        int cum_offset    = 0;
        int index         = 0;
        for (int i = 0; i < batch_size; i++) {
            const int seq_len = sequence_length[i];
            for (int j = 0; j < seq_len; j++) {
                tmp_mask_offset[index] = cum_offset;
                index++;
            }
            cum_offset += max_seq_len - seq_len;
            total_seq_len += seq_len;
        }
        valid_word_num[0] = total_seq_len;
    }
}

template<typename T>
void invokeGetPaddingOffsetFromMask(int*         h_token_num,
                                    int*         d_token_num,
                                    int*         tmp_mask_offset,
                                    const T*     mask,
                                    const int    batch_size,
                                    const int    max_seq_len,
                                    const int    ld_mask,
                                    cudaStream_t stream)
{
    getPaddingOffsetFromMaskKernel<<<1, 256, sizeof(int) * batch_size, stream>>>(
        d_token_num, tmp_mask_offset, mask, batch_size, max_seq_len, ld_mask);
    CUDA_CHECK(cudaMemcpyAsync(h_token_num, d_token_num, sizeof(int), cudaMemcpyDeviceToHost, stream));
}

template<typename T>
__global__ void remove_padding(T* tgt, const T* src, const int* padding_offset, const int n)
{
    const int tid        = threadIdx.x;
    const int bid        = blockIdx.x;
    const int src_seq_id = bid + padding_offset[bid];
    const int tgt_seq_id = bid;

    for (int i = tid; i < n; i += blockDim.x) {
        tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
    }
}

template<typename T>
void invokeRemovePadding(
    T* dst, const T* src, const int* padding_offset, const int token_num, const int hidden_dim, cudaStream_t stream)
{
    remove_padding<<<token_num, 256, 0, stream>>>(dst, src, padding_offset, hidden_dim);
}

template<typename T>
__global__ void rebuild_sequence_length_padding(const T* src, T* dst, const int* padding_offset, const int n)
{
    const int tid        = threadIdx.x;
    const int bid        = blockIdx.x;
    const int dst_seq_id = bid + padding_offset[bid];
    const int src_seq_id = bid;

    for (int i = tid; i < n; i += blockDim.x) {
        dst[dst_seq_id * n + i] = src[src_seq_id * n + i];
    }
}

template<typename T>
void invokeRebuildPadding(
    T* dst, const T* src, const int* padding_offset, const int m, const int n, cudaStream_t stream)
{
    // src: [token_num, hidden_dim]
    // dst: [batch_size*max_seq_len, hidden_dim]
    rebuild_sequence_length_padding<<<m, 256, 0, stream>>>(src, dst, padding_offset, n);
}

__global__ void getTrtPaddingOffsetKernel(int* trt_mha_padding_offset, const int* sequence_length, const int batch_size)
{
    // use for get tensorrt fused mha padding offset
    // when we remove the padding

    extern __shared__ int tmp_offset[];
    if (threadIdx.x == 0) {
        tmp_offset[0] = 0;
        for (int i = 0; i < batch_size; i++) {
            tmp_offset[i + 1] = tmp_offset[i] + sequence_length[i];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < batch_size + 1; i += blockDim.x) {
        trt_mha_padding_offset[i] = tmp_offset[i];
    }
}

void invokeGetTrtPaddingOffset(int*         trt_mha_padding_offset,
                               const int*   sequence_length,
                               const int    batch_size,
                               cudaStream_t stream)
{
    getTrtPaddingOffsetKernel<<<1, 256, sizeof(int) * (batch_size + 1), stream>>>(
        trt_mha_padding_offset, sequence_length, batch_size);
}

template<typename T>
__global__ void getTrtPaddingOffsetKernelFromMask(int*      trt_mha_padding_offset,
                                                  const T*  mask,
                                                  const int batch_size,
                                                  const int max_seq_len,
                                                  const int ld_mask)
{
    // use for get tensorrt fused mha padding offset
    // when we remove the padding

    extern __shared__ int sequence_length[];
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        const T* b_mask = mask + i * ld_mask;
        int len = 0;
        for (int j = 0; j < max_seq_len; ++j) {
            if ((float(b_mask[j]) - 0.0) > -1e-5 && (float(b_mask[j]) - 0.0) < 1e-5) {
                ++len;
            } else {
                break;
            }
        }
        sequence_length[i] = len;
    }
    __syncthreads();

    int *tmp_offset = sequence_length + batch_size;
    if (threadIdx.x == 0) {
        tmp_offset[0] = 0;
        for (int i = 0; i < batch_size; i++) {
            tmp_offset[i + 1] = tmp_offset[i] + sequence_length[i];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < batch_size + 1; i += blockDim.x) {
        trt_mha_padding_offset[i] = tmp_offset[i];
    }
}

template<typename T>
void invokeGetTrtPaddingOffsetFromMask(int*         trt_mha_padding_offset,
                                       const T*     mask,
                                       const int    batch_size,
                                       const int    max_seq_len,
                                       const int    ld_mask,
                                       cudaStream_t stream)
{
    getTrtPaddingOffsetKernelFromMask<<<1, 256, sizeof(int) * (2*batch_size + 1), stream>>>(
        trt_mha_padding_offset, mask, batch_size, max_seq_len, ld_mask);
}

Status CudaEffectiveTransformerLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<EffectiveTransformerLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer_param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: layer_param is nil");
    }

    if ((layer_param->is_remove_padding && inputs.size() <= 2) ||
        (!layer_param->is_remove_padding && (inputs.size() != 2 && inputs.size() != 3))) {
        LOGE("Error: input number not support\n");
        return Status(TNNERR_LAYER_ERR, "Error: input number not support");
    }

    Blob* input_blob1 = inputs[0];
    Blob* input_blob2 = inputs[1];
    auto input_dims1 = input_blob1->GetBlobDesc().dims;
    auto input_dims2 = input_blob2->GetBlobDesc().dims;
    void* input_data1 = input_blob1->GetHandle().base;

    if (input_dims1.size() != 3) {
        LOGE("Error: input dims not support\n");
        return Status(TNNERR_LAYER_ERR, "Error: input dims not support");
    }

    int type_size = DataTypeUtils::GetBytesSize(input_blob1->GetBlobDesc().data_type);
    int data_count = DimsVectorUtils::Count(input_dims1);

    auto stream = context_->GetStream();
    if (layer_param->is_remove_padding) {
        if (outputs.size() != 3) {
            LOGE("Error: output number not support\n");
            return Status(TNNERR_LAYER_ERR, "Error: output number not support");
        }
        Blob* dense_blob = outputs[0];
        Blob* offset_blob = outputs[1];
        Blob* trt_offset_blob = outputs[2];
        void* dense_data = dense_blob->GetHandle().base;
        int* offset_data = reinterpret_cast<int *>(offset_blob->GetHandle().base);
        int* trt_offset_data = reinterpret_cast<int *>(trt_offset_blob->GetHandle().base);

        int ld_mask = DimsVectorUtils::Count(input_blob2->GetBlobDesc().dims, 1);

        int h_token_num = 0;
        if (input_blob1->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            float* input_data2 = reinterpret_cast<float *>(input_blob2->GetHandle().base);
            invokeGetPaddingOffsetFromMask(&h_token_num, offset_data, offset_data + 1, input_data2, input_dims1[0], input_dims1[1], ld_mask, stream);
        } else {
            half* input_data2 = reinterpret_cast<half *>(input_blob2->GetHandle().base);
            invokeGetPaddingOffsetFromMask(&h_token_num, offset_data, offset_data + 1, input_data2, input_dims1[0], input_dims1[1], ld_mask, stream);
        }
        if (h_token_num < 0) {
            LOGE("Error: token num is invalid\n");
            return Status(TNNERR_LAYER_ERR, "Error: token num is invalid");
        }

        cudaMemsetAsync(dense_data, 0, data_count * type_size, stream);
        if (h_token_num > 0) {
            if (input_blob1->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
                invokeRemovePadding(reinterpret_cast<float*>(dense_data), reinterpret_cast<float*>(input_data1), offset_data + 1, h_token_num, input_dims1[2], stream);
            } else if (input_blob1->GetBlobDesc().data_type == DATA_TYPE_HALF) {
                invokeRemovePadding(reinterpret_cast<half*>(dense_data), reinterpret_cast<half*>(input_data1), offset_data + 1, h_token_num, input_dims1[2], stream);
            } else {
                LOGE("Error: input dtype not support\n");
                return Status(TNNERR_LAYER_ERR, "Error: input dtype not support");
            }
        }
        auto& info_map = context_->GetExtraInfoMap();
        info_map["int_transformer_runtime_token_num"] = make_any<int>(h_token_num);

        if (input_blob1->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            float* input_data2 = reinterpret_cast<float *>(input_blob2->GetHandle().base);
            invokeGetTrtPaddingOffsetFromMask(trt_offset_data, input_data2, input_dims1[0], input_dims1[1], ld_mask, stream);
        } else {
            half* input_data2 = reinterpret_cast<half *>(input_blob2->GetHandle().base);
            invokeGetTrtPaddingOffsetFromMask(trt_offset_data, input_data2, input_dims1[0], input_dims1[1], ld_mask, stream);
        }
    } else {
        if (outputs.size() != 1 && outputs.size() != 2) {
            LOGE("Error: output number not support\n");
            return Status(TNNERR_LAYER_ERR, "Error: output number not support");
        }
        Blob* output_blob = outputs[0];
        void* output_data = output_blob->GetHandle().base;
        int* input_data2 = reinterpret_cast<int *>(input_blob2->GetHandle().base);

        int rt_token_num = 0;
        auto& info_map = context_->GetExtraInfoMap();
        if (info_map.find("int_transformer_runtime_token_num") != info_map.end()) {
            rt_token_num = any_cast<int>(info_map["int_transformer_runtime_token_num"]); 
        }
        if (rt_token_num < 0) {
            LOGE("Error: token num is invalid\n");
            return Status(TNNERR_LAYER_ERR, "Error: token num is invalid");
        }

        cudaMemsetAsync(output_data, 0, data_count * type_size, stream);
        if (rt_token_num > 0) {
            if (input_blob1->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
                invokeRebuildPadding(reinterpret_cast<float*>(output_data), reinterpret_cast<float*>(input_data1), input_data2 + 1, rt_token_num, input_dims1[2], stream);
            } else if (input_blob1->GetBlobDesc().data_type == DATA_TYPE_HALF) {
                invokeRebuildPadding(reinterpret_cast<half*>(output_data), reinterpret_cast<half*>(input_data1), input_data2 + 1, rt_token_num, input_dims1[2], stream);
            } else {
                LOGE("Error: input dtype not support\n");
                return Status(TNNERR_LAYER_ERR, "Error: input dtype not support");
            }
        }
        info_map["int_transformer_runtime_token_num"] = make_any<int>(-1);
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(EffectiveTransformer, LAYER_EFFECTIVE_TRANSFORMER);

}  // namespace TNN_NS
