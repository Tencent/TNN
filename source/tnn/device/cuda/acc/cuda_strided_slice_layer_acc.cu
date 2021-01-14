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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(StrideSlice, LAYER_STRIDED_SLICE);

template<int THREAD_PER_BLOCK, int ELE_PER_THREAD>
__global__ void strided_slice_kernel(int size, const float * __restrict__ srcData, int input_c, int input_h,
        int input_w, const int* __restrict__ begin, const int* __restrict__ strides, float* __restrict__ dstData, 
        int output_c, int output_h, int output_w, int div_c, int div_n) {
    int block_offset = blockIdx.x * THREAD_PER_BLOCK * ELE_PER_THREAD;

    const int mul_n = input_c * input_h * input_w * strides[3];
    const int mul_c = input_h * input_w * strides[2];
    const int mul_h = input_w * strides[1];
    const int mul_w = strides[0];
    const int offset = begin[3] * input_c * input_h * input_w +
                   + begin[2] * input_h * input_w +
                   + begin[1] * input_w 
                   + begin[0];

    #pragma unroll
    for(int i =0;i < ELE_PER_THREAD ;i++) {
        int index = block_offset + i * THREAD_PER_BLOCK + threadIdx.x;
        if (index < size) {
            int w = index % output_w;
            int h = index / output_w % output_h;
            int c = index / div_c % output_c;
            int n = index / div_n ;
            int input_index = n * mul_n + c * mul_c + h * mul_h + w * mul_w + offset;
            dstData[index] = srcData[input_index];
        }
    }
}

Status CudaStrideSliceLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }    
    CreateTempBuf(4 * sizeof(int));
    CreateTempBuf(4 * sizeof(int));
    auto params = dynamic_cast<StrideSliceLayerParam *>(param);
    if (!params) {
        LOGE("Error: ShuffleLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: ShuffleLayerParam is nil");
    }

    cudaMemcpyAsync(tempbufs_[0].ptr, &(params->begins[0]), 4 * sizeof(int), cudaMemcpyHostToDevice, context_->GetStream());
    cudaMemcpyAsync(tempbufs_[1].ptr, &(params->strides[0]), 4 * sizeof(int), cudaMemcpyHostToDevice, context_->GetStream());

    return TNN_OK;
}

Status CudaStrideSliceLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaStrideSliceLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];

    auto input_dims   = input_blob->GetBlobDesc().dims;
    int input_n = input_dims[0];
    int input_c = input_dims[1];
    int input_h = input_dims[2];
    int input_w = input_dims[3];

    auto output_dims = output_blob->GetBlobDesc().dims;
    int output_c = output_dims[1];
    int output_h = output_dims[2];
    int output_w = output_dims[3];
    int div_c = output_w * output_h;
    int div_n = output_w * output_h * output_c;

    int count = DimsVectorUtils::Count(output_dims);

    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);

    const int THREAD_PER_BLOCK = 128;
    const int ELE_PER_THREAD = 64;
    int blocks = (count + THREAD_PER_BLOCK * ELE_PER_THREAD - 1) / (THREAD_PER_BLOCK * ELE_PER_THREAD);
    strided_slice_kernel<THREAD_PER_BLOCK, ELE_PER_THREAD><<<blocks, THREAD_PER_BLOCK, 0, context_->GetStream()>>>(
        count, input_data, input_c, input_h, input_w, (const int*)tempbufs_[0].ptr, (const int*)tempbufs_[1].ptr,
        output_data, output_c, output_h, output_w, div_c, div_n);
    return TNN_OK;
}

REGISTER_CUDA_ACC(StrideSlice, LAYER_STRIDED_SLICE);

}  // namespace TNN_NS