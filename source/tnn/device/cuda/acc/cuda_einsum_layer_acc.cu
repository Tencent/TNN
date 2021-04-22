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

DECLARE_CUDA_ACC(Einsum, LAYER_EINSUM);

__global__ void einsum_permute_kernel(int n, const float *srcData, int num_axes, int *permute_order,
        int *old_steps, int *new_steps, float *dstData) {
    CUDA_KERNEL_LOOP(index, n) {
        int old_idx = 0;
        int idx = index;
        for (int j = 0; j < num_axes; ++j) {
            int order = permute_order[j];
            old_idx += (idx / new_steps[j]) * old_steps[order];
            idx %= new_steps[j];
        }
        dstData[index] = srcData[old_idx];
    }
}

__global__ void einsum_sum_kernel(const float* in, float* out, int outer_count, int reducer_count, int inner_count) {
    CUDA_KERNEL_LOOP(index, outer_count) {
        const float* input_data = in + index * inner_count * reducer_count;
        float* output_data = out + index * inner_count;
        for (int c = 0; c < reducer_count; c++) {
            for (int ic = 0; ic < inner_count; ic++) {
                output_data[ic] += input_data[ic];
            }
            input_data += inner_count;
        }
    }
}

__global__ void einsum_mul_kernel(int count, const float* in1, const float* in2, float* out,
        const int* input1_dims, int input1_dims_size, const int* input2_dims, int input2_dims_size,
        const int* output_dims, int output_dim_size) {
    CUDA_KERNEL_LOOP(index, count) {
        int diff1 = output_dim_size - input1_dims_size;
        int diff2 = output_dim_size - input2_dims_size;
        int input1_offset = 0;
        int input2_offset = 0;
        int prod = count;
        for (int i = 0; i < diff1; i++) {
            prod /= output_dims[i];
        }
        for (int i = 0; i < input1_dims_size; i++) {
            prod /= output_dims[i+diff1];
            int mod = index / prod % output_dims[i+diff1];
            mod = min(mod, input1_dims[i]-1);
            input1_offset = input1_offset * input1_dims[i] + mod;
        }
        prod = count;
        for (int i = 0; i < diff2; i++) {
            prod /= output_dims[i];
        }
        for (int i = 0; i < input2_dims_size; i++) {
            prod /= output_dims[i+diff2];
            int mod = index / prod % output_dims[i+diff2];
            mod = min(mod, input2_dims[i]-1);
            input2_offset = input2_offset * input2_dims[i] + mod;
        }
        out[index] = in1[input1_offset] * in2[input2_offset];
    }
}

template<int THREAD_PER_BLOCK>
__global__ void einsum_dot_kernel(const float* a, const float* b, float* c, int count) {
    __shared__ double ssum[THREAD_PER_BLOCK/32];
    double thread_sum = 0.f;
    for (int i = threadIdx.x; i < count; i+=THREAD_PER_BLOCK) {
        thread_sum += a[i] * b[i];
    }
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, 16, 32);
    thread_sum += __shfl_down_sync(0x0000ffff, thread_sum, 8, 16);
    thread_sum += __shfl_down_sync(0x000000ff, thread_sum, 4, 8);
    thread_sum += __shfl_down_sync(0x0000000f, thread_sum, 2, 4);
    thread_sum += __shfl_down_sync(0x00000003, thread_sum, 1, 2);

    if (threadIdx.x % 32 == 0) {
        ssum[threadIdx.x / 32] = thread_sum;
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x / 32) {
        thread_sum = ssum[threadIdx.x];
    } else {
        thread_sum = 0;
    }

    thread_sum += __shfl_down_sync(0x0000000f, thread_sum, 2, 4);
    thread_sum += __shfl_down_sync(0x00000003, thread_sum, 1, 2);

    if (threadIdx.x == 0) {
        c[0] = thread_sum;
    }
}

std::shared_ptr<Blob> EinsumPermute(Blob *input_blob, const std::vector<int> &orders,
        void* buf1, void* buf2, void* buf3, cudaStream_t stream) {
    auto output_blob_ptr = std::make_shared<Blob>(input_blob->GetBlobDesc(), true);
    auto *output_blob    = output_blob_ptr.get();
    auto input_dims      = input_blob->GetBlobDesc().dims;
    auto output_dims     = input_blob->GetBlobDesc().dims;
    const int dims_size  = input_dims.size();
    for (int i = 0; i < dims_size; i++) {
        output_dims[i] = input_dims[orders[i]];
    }
    output_blob->GetBlobDesc().dims = output_dims;

    std::vector<int> input_step;
    std::vector<int> output_step;
    for (int i = 0; i < input_dims.size(); ++i) {
        input_step.push_back(DimsVectorUtils::Count(input_dims, i + 1));
        output_step.push_back(DimsVectorUtils::Count(output_dims, i + 1));
    }
    cudaMemcpyAsync(buf1, orders.data(), orders.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buf2, input_step.data(), input_dims.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buf3, output_step.data(), input_dims.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream);

    float *input_data   = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data  = static_cast<float *>(output_blob->GetHandle().base);
    const int count = DimsVectorUtils::Count(output_dims);
    einsum_permute_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, stream>>>(count,
        input_data, output_dims.size(), (int*)buf1, (int*)buf2, (int*)buf3, output_data);
    return output_blob_ptr;
}

void EinsumSqueeze(Blob *input_blob, const int axis) {
    auto output_dims = input_blob->GetBlobDesc().dims;
    output_dims.erase(output_dims.begin() + axis);
    input_blob->GetBlobDesc().dims = output_dims;
}

std::shared_ptr<Blob> EinsumSum(Blob *input_blob, const int axis, cudaStream_t stream) {
    const auto input_desc = input_blob->GetBlobDesc();
    auto output_desc      = input_desc;
    auto output_dims      = output_desc.dims;
    output_dims.erase(output_dims.begin() + axis);
    output_desc.dims     = output_dims;
    auto output_blob_ptr = std::make_shared<Blob>(output_desc, true);
    auto *output_blob    = output_blob_ptr.get();

    auto input_dims   = input_blob->GetBlobDesc().dims;
    int outer_count   = DimsVectorUtils::Count(input_dims, 0, axis);
    int reducer_count = input_dims[axis];
    int inner_count   = DimsVectorUtils::Count(input_dims, axis + 1);
    inner_count       = inner_count == 0 ? 1 : inner_count;
    input_dims[axis]  = 1;

    float *input_data      = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data     = static_cast<float *>(output_blob->GetHandle().base);
    einsum_sum_kernel<<<TNN_CUDA_GET_BLOCKS(outer_count), TNN_CUDA_NUM_THREADS, 0, stream>>>(input_data, output_data,
        outer_count, reducer_count, inner_count);
    return output_blob_ptr;
}

std::shared_ptr<Blob> EinsumMul(Blob *a, Blob *b, void* buf1, void* buf2, void* buf3, cudaStream_t stream) {
    std::vector<void *> input_ptrs       = {a->GetHandle().base, b->GetHandle().base};
    auto output_dims                     = DimsVectorUtils::Max(a->GetBlobDesc().dims, b->GetBlobDesc().dims);
    auto output_desc                     = a->GetBlobDesc();
    output_desc.dims                     = output_dims;
    auto output_ptr                      = std::make_shared<Blob>(output_desc, true);
    auto *output                         = output_ptr.get();
    int input1_dims_size = a->GetBlobDesc().dims.size();
    int input2_dims_size = b->GetBlobDesc().dims.size();

    cudaMemcpyAsync(buf1, a->GetBlobDesc().dims.data(), input1_dims_size*sizeof(int),
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buf2, b->GetBlobDesc().dims.data(), input2_dims_size*sizeof(int),
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buf3, output_dims.data(), output_dims.size()*sizeof(int),
        cudaMemcpyHostToDevice, stream);

    int count = DimsVectorUtils::Count(output_dims);
    einsum_mul_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, stream>>>(count,
        (const float*)(a->GetHandle().base), (const float*)(b->GetHandle().base), (float*)(output->GetHandle().base),
        (const int*)buf1, input1_dims_size, (const int*)buf2, input2_dims_size, (const int*)buf3, output_dims.size());
    return output_ptr;
}

void EinsumFlatten(Blob *input_blob) {
    const int output_dims_size     = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims);
    input_blob->GetBlobDesc().dims = {output_dims_size};
}

std::shared_ptr<Blob> EinsumDot(Blob *a, Blob *b, cudaStream_t stream) {
    auto output_blob_desc = a->GetBlobDesc();
    output_blob_desc.dims = {1};
    auto output_blob_ptr  = std::make_shared<Blob>(output_blob_desc, true);
    auto *output_blob     = output_blob_ptr.get();
    const int data_size   = a->GetBlobDesc().dims[0];
    float *a_data         = static_cast<float *>(a->GetHandle().base);
    float *b_data         = static_cast<float *>(b->GetHandle().base);
    float *output_data    = static_cast<float *>(output_blob->GetHandle().base);
    einsum_dot_kernel<128><<<1, 128, 0, stream>>>(a_data, b_data, output_data, data_size);
    return output_blob_ptr;
}

Status CudaEinsumLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    CreateTempBuf(6 * sizeof(int));
    CreateTempBuf(6 * sizeof(int));
    CreateTempBuf(6 * sizeof(int));
    CreateTempBuf(6 * sizeof(int));
    CreateTempBuf(6 * sizeof(int));
    CreateTempBuf(6 * sizeof(int));
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaEinsumLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaEinsumLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<EinsumLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: EinsumLayerParam is nil");
    }

    std::vector<std::shared_ptr<Blob>> permuted_operands;
    for (int i = 0; i < inputs.size(); i++) {
        //auto operand = inputs[i];
        auto operand_ptr          = std::make_shared<Blob>(inputs[i]->GetBlobDesc(), inputs[i]->GetHandle());
        auto *operand             = operand_ptr.get();
        operand->GetBlobDesc().dims = param->operand_dims[i];
        permuted_operands.push_back(EinsumPermute(operand, param->perm_shapes[i], tempbufs_[0].ptr,
            tempbufs_[1].ptr, tempbufs_[2].ptr, context_->GetStream()));
    }

    int out_size = param->out_size;
    int perm_index = param->dim_last_op.size();
    auto dim_last_op = param->dim_last_op;
    auto result = permuted_operands[0];

    if (param->has_zero_size_dim) {
        std::vector<int> out_shape(out_size);
        int output_shape_count = 1;
        for (int i = 0; i < out_size; i++) {
            out_shape[i] = permuted_operands[dim_last_op[i]].get()->GetBlobDesc().dims[i];
            output_shape_count *= out_shape[i];
        }
        float *output_ptr = static_cast<float *>(outputs[0]->GetHandle().base);
        cudaMemset(output_ptr, 0, sizeof(float) * output_shape_count);

        return TNN_OK;
    }

    int dim = out_size;
    for (int i = dim; i < perm_index; ++i, ++dim) {
        if (dim_last_op[i] == 0) {
            if (result.get()->GetBlobDesc().dims[dim] == 1) {
                EinsumSqueeze(result.get(), dim--);
            } else {
                result = EinsumSum(result.get(), dim--, context_->GetStream());
            }
        }
    }

    auto operand = permuted_operands[1];
    std::vector<int> sum_dims;

    dim = out_size;
    for (int j = dim; j < perm_index; ++j, ++dim) {
        if (dim_last_op[j] < 1) {
            EinsumSqueeze(operand.get(), dim);
            --dim;
        } else if (dim_last_op[j] == 1) {
            if (result.get()->GetBlobDesc().dims[dim] == 1) {
                operand = EinsumSum(operand.get(), dim, context_->GetStream());
                EinsumSqueeze(result.get(), dim);
                --dim;
            } else {
                sum_dims.push_back(dim);
            }
        }
    }

    if (sum_dims.empty()) {
        result = EinsumMul(result.get(), operand.get(), tempbufs_[0].ptr,
            tempbufs_[1].ptr, tempbufs_[2].ptr, context_->GetStream());
    } else if (sum_dims.size() == result.get()->GetBlobDesc().dims.size()) {
        EinsumFlatten(result.get());
        EinsumFlatten(operand.get());
        result = EinsumDot(result.get(), operand.get(), context_->GetStream());
    } else {
        result = EinsumMul(result.get(), operand.get(), tempbufs_[0].ptr,
            tempbufs_[1].ptr, tempbufs_[2].ptr, context_->GetStream());
        for (const auto axis : sum_dims) {
            result = EinsumSum(result.get(), axis, context_->GetStream());
        }
    }

    const int data_count = DimsVectorUtils::Count(result.get()->GetBlobDesc().dims);
    cudaMemcpyAsync(outputs[0]->GetHandle().base, result.get()->GetHandle().base,
        data_count * sizeof(float), cudaMemcpyDeviceToDevice, context_->GetStream());
//    context_->Synchronize();

    return TNN_OK;
}

REGISTER_CUDA_ACC(Einsum, LAYER_EINSUM);

}  // namespace TNN_NS
