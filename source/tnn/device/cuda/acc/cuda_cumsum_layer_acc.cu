// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

namespace TNN_NS {

DECLARE_CUDA_ACC(Cumsum, LAYER_CUMSUM);

Status CudaCumsumLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);;
}

Status CudaCumsumLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

template<typename T>
__global__ void cumsum_kernel(const T* input, T* output, const int dim_curr, const int dim_post) {
    int offset = blockIdx.x * dim_curr * dim_post + (blockIdx.y * blockDim.x + threadIdx.x);
    T curr_cumsum = T(0);
    for (int i=0; i<dim_curr; i++) {
        curr_cumsum += input[offset];
        output[offset] = curr_cumsum;
        offset += dim_post;
    }
}

template<typename T>
__global__ void cumsum_kernel_reversed(const T* input, T* output, const int dim_curr, const int dim_post) {
    int offset = blockIdx.x * dim_curr * dim_post + (blockIdx.y * blockDim.x + threadIdx.x) + dim_post * (dim_curr - 1);
    T curr_cumsum = T(0);
    for (int i=0; i<dim_curr; i++) {
        curr_cumsum += input[offset];
        output[offset] = curr_cumsum;
        offset -= dim_post;
    }
}

template<typename T>
__global__ void cumsum_kernel_exclusive(const T* input, T* output, const int dim_curr, const int dim_post) {
    int offset = blockIdx.x * dim_curr * dim_post + (blockIdx.y * blockDim.x + threadIdx.x);
    T curr_cumsum = T(0);
    for (int i=0; i<dim_curr; i++) {
        output[offset] = curr_cumsum;
        curr_cumsum += input[offset];
        offset += dim_post;
    }
}

template<typename T>
__global__ void cumsum_kernel_exclusive_reversed(const T* input, T* output, const int dim_curr, const int dim_post) {
    int offset = blockIdx.x * dim_curr * dim_post + (blockIdx.y * blockDim.x + threadIdx.x) + dim_post * (dim_curr - 1);
    T curr_cumsum = T(0);
    for (int i=0; i<dim_curr; i++) {
        output[offset] = curr_cumsum;
        curr_cumsum += input[offset];
        offset -= dim_post;
    }
}

template <typename T>
void CudaCumsumForwardImpl(Blob* input_blob, Blob* output_blob, const dim3 blocks,
                           const int thread_per_block, const int dim_curr, const int dim_post,
                           const bool exclusive, const bool reverse, cudaStream_t& stream) {
    T* input_data  = reinterpret_cast<T *>(input_blob->GetHandle().base);
    T* output_data = reinterpret_cast<T *>(output_blob->GetHandle().base);
    if (exclusive) {
        if (reverse) {
            cumsum_kernel_exclusive_reversed<T><<<blocks, thread_per_block, 0, stream>>>
                (input_data, output_data, dim_curr, dim_post);
        } else {
            cumsum_kernel_exclusive<T><<<blocks, thread_per_block, 0, stream>>>
                (input_data, output_data, dim_curr, dim_post);
        }
    } else {
        if (reverse) {
            cumsum_kernel_reversed<T><<<blocks, thread_per_block, 0, stream>>>
                (input_data, output_data, dim_curr, dim_post);
        } else {
            cumsum_kernel<T><<<blocks, thread_per_block, 0, stream>>>
                (input_data, output_data, dim_curr, dim_post);
        }
    }
}


Status CudaCumsumLayerAcc::Forward(const std::vector<Blob*> &inputs, const std::vector<Blob*> &outputs) {
    // Operator Cumsum input.dim == output.dim
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];
    auto input_dims  = input_blob->GetBlobDesc().dims;

    auto cumsum_param  = dynamic_cast<CumsumLayerParam*>(param_);
    if (cumsum_param == nullptr) {
        LOGE("Error: CudaCumsumLayer forward load layer param failed\n");
        return Status(TNNERR_MODEL_ERR, "Error: CudaCumsumLayer forward Load layer param failed!");
    }
    if (cumsum_param->axis < 0) {
        cumsum_param->axis += input_dims.size();
    }

    int dim_pre  = 1;
    int dim_curr = input_dims[cumsum_param->axis];
    int dim_post = 1;
    for (int i=0; i<cumsum_param->axis; i++) {
        dim_pre *= input_dims[i];
    }
    for (int i=cumsum_param->axis+1; i<input_dims.size(); i++) {
        dim_post *= input_dims[i];
    }

    const int THREAD_PER_BLOCK = 128;
    dim3 blocks;
    blocks.x = dim_pre;
    blocks.y = (dim_post + THREAD_PER_BLOCK - 1 ) / THREAD_PER_BLOCK;
    if (blocks.x > 65535 || blocks.y > 65535) {
        LOGE("Error: CudaCumsumLayer forward layer cuda block.x or block.y > 65535, large kernel not supported yet.\n");
        return Status(TNNERR_MODEL_ERR, "Error: CudaCumsumLayer forward layer cuda block.x or block.y > 65535, large kernel not supported yet.");
    }

    // Run cuda Kernel
    auto data_type = output_blob->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_FLOAT) {
        CudaCumsumForwardImpl<float>(input_blob, output_blob, blocks, THREAD_PER_BLOCK, dim_curr, dim_post,
                                     cumsum_param->exclusive, cumsum_param->reverse, context_->GetStream());
    } else if (data_type == DATA_TYPE_HALF) {
        CudaCumsumForwardImpl<__half>(input_blob, output_blob, blocks, THREAD_PER_BLOCK, dim_curr, dim_post,
                                      cumsum_param->exclusive, cumsum_param->reverse, context_->GetStream());
    } else if (data_type == DATA_TYPE_INT32) {
        CudaCumsumForwardImpl<int>(input_blob, output_blob, blocks, THREAD_PER_BLOCK, dim_curr, dim_post,
                                   cumsum_param->exclusive, cumsum_param->reverse, context_->GetStream());
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(Cumsum, LAYER_CUMSUM);

}  // namespace TNN_NS
