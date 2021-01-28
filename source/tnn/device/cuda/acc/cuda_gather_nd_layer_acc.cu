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

DECLARE_CUDA_ACC(GatherND, LAYER_GATHERND);

__global__ void gather_nd_kernel(int count, const float* data, const int* indices, const int* input_stride, int slice_size, float* dst) {
    CUDA_KERNEL_LOOP(idx, count) {
        const int *indices_ptr = indices + idx * slice_size;
        int input_index = 0;
        for (int i=0; i<slice_size; i++) {
            input_index += indices_ptr[i] * input_stride[i];
        }
        dst[idx] = data[input_index]; 
    }
}

Status CudaGatherNDLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if(ret != TNN_OK) {
        return ret;
    }
    auto layer_param = dynamic_cast<GatherNDLayerParam*>(param);
    CHECK_PARAM_NULL(layer_param);
    int batch_dims = layer_param->batch_dims;
    
    if (batch_dims != 0) {
        return Status(TNNERR_PARAM_ERR, "GatherNDLayerParam has invalid param batch_dims");
    }

    auto input_dims = (*(inputs.begin()))->GetBlobDesc().dims;
    CreateTempBuf(input_dims.size() * sizeof(int));
    return TNN_OK;
}
   
Status CudaGatherNDLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaGatherNDLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_data_blob  = inputs[0];
    Blob *indices_blob  = inputs[1];
    Blob *output_blob = outputs[0];


    auto input_data_dims = input_data_blob->GetBlobDesc().dims;
    auto indices_dims = indices_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;
    int count = DimsVectorUtils::Count(output_dims);
    auto input_stride = DimsVectorUtils::StrideOfShape(input_data_dims);
    if (indices_dims[indices_dims.size()-1] != input_data_dims.size()) {
        return Status(TNNERR_PARAM_ERR, "GatherNDLayerParam has invalid param indices_dims");
    }
    cudaMemcpyAsync(tempbufs_[0].ptr, input_stride.data(), input_stride.size() * sizeof(int), cudaMemcpyHostToDevice, context_->GetStream());

    const int slice_size = indices_dims[indices_dims.size()-1];
    float* input_data = static_cast<float*>(input_data_blob->GetHandle().base);
    int* indices_data = static_cast<int*>(indices_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);

    gather_nd_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
        count, input_data, indices_data, (int*)tempbufs_[0].ptr, slice_size, output_data);    
    return TNN_OK;
}

REGISTER_CUDA_ACC(GatherND, LAYER_GATHERND);

}  // namespace TNN_NS
