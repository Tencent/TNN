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

DECLARE_CUDA_ACC(SplitV, LAYER_SPLITV);

template<int THREAD_PER_BLOCK, int ELE_PER_THREAD>
__global__ void splitv_separate_kernel(
    const float * __restrict__ src, float * dst,
    const int inner_size, const int in_stride, 
    const int split_start, const int split_end)
{
    int block_offset = blockIdx.x * THREAD_PER_BLOCK * ELE_PER_THREAD;

    const int split_size = split_end - split_start;
    const int size = split_size * inner_size;
    src += (blockIdx.z * gridDim.y + blockIdx.y) * in_stride;
    dst += (blockIdx.z * gridDim.y + blockIdx.y) * size;
  
    #pragma unroll
    for(int i =0;i < ELE_PER_THREAD ;i++)
    {
        int index = block_offset + i * THREAD_PER_BLOCK + threadIdx.x;
        if (index < size){
          int input_index = index + split_start * inner_size;
          dst[index] = __ldg(src + input_index);
        }
    }

}

Status CudaSplitVLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    auto layer_param = dynamic_cast<SplitVLayerParam *>(param_);
    if (!layer_param || layer_param->slices.size() != outputs.size()) {
        return Status(TNNERR_PARAM_ERR, "CudaSplitVLayerAcc has invalid param, slices size != output blobs size");
    }

    return TNN_OK;
}

Status CudaSplitVLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaSplitVLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<SplitVLayerParam *>(param_);
    int axis = layer_param->axis;
    Blob *input_blob  = inputs[0];
    auto dims = input_blob->GetBlobDesc().dims;
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);

    const int THREAD_PER_BLOCK = 128;
    const int ELE_PER_THREAD = 16;

    const int in_stride = DimsVectorUtils::Count(dims, axis);
    const int inner_size = DimsVectorUtils::Count(dims, axis + 1);
    
    auto slices = layer_param->slices;
    int split_num = slices.size();

    int split_begin = 0;
    for(int i= 0; i < split_num; i++) {
      if (slices[i] > 0) {
        Blob* output_blob = outputs[i];
        int split_end = split_begin + slices[i];
        dim3 griddim;
        griddim.x = (slices[i] * inner_size + ELE_PER_THREAD * THREAD_PER_BLOCK - 1) / (ELE_PER_THREAD * THREAD_PER_BLOCK);
        griddim.y = DimsVectorUtils::Count(dims, 1, axis);
        griddim.z = DimsVectorUtils::Count(dims, 0, min(1, axis));

        float* output_data = static_cast<float*>(output_blob->GetHandle().base);
        splitv_separate_kernel<THREAD_PER_BLOCK, ELE_PER_THREAD><<<griddim, THREAD_PER_BLOCK, 0, context_->GetStream()>>>
            (input_data, output_data, inner_size, in_stride, split_begin, split_end);
        split_begin = split_end;
      }
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(SplitV, LAYER_SPLITV);

}  // namespace TNN_NS
