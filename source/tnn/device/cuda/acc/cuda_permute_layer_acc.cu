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

DECLARE_CUDA_ACC(Permute, LAYER_PERMUTE);

__global__ void permute_kernel(int n, const float *srcData, int num_axes, int *permute_order,
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

Status CudaPermuteLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }
    auto params = dynamic_cast<PermuteLayerParam *>(param);
    if (!params) {
        return Status(TNNERR_MODEL_ERR, "Error: PermuteLayerParam is empyt");
    }
    Blob *input_blob = inputs[0];
    Blob *output_blob = outputs[0];
    auto input_dims = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;
    ASSERT(input_dims.size() == output_dims.size());

    CreateTempBuf(input_dims.size() * sizeof(int));
    CreateTempBuf(input_dims.size() * sizeof(int));
    CreateTempBuf(input_dims.size() * sizeof(int));
    cudaMemcpyAsync(tempbufs_[0].ptr, &(params->orders[0]), 4 * sizeof(int), cudaMemcpyHostToDevice, context_->GetStream());

    std::vector<int> input_step;
    std::vector<int> output_step;
    for (int i = 0; i < input_dims.size(); i++) {
        input_step.push_back(DimsVectorUtils::Count(input_dims, i + 1));
        output_step.push_back(DimsVectorUtils::Count(output_dims, i + 1));
    }
    cudaMemcpyAsync(tempbufs_[1].ptr, &(input_step[0]), input_dims.size() * sizeof(int), cudaMemcpyHostToDevice, context_->GetStream());
    cudaMemcpyAsync(tempbufs_[2].ptr, &(output_step[0]), input_dims.size() * sizeof(int), cudaMemcpyHostToDevice, context_->GetStream());
    return TNN_OK;
}

Status CudaPermuteLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaPermuteLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    int count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    permute_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
        count, input_data, 4, (int*)tempbufs_[0].ptr, (int*)tempbufs_[1].ptr, (int*)tempbufs_[2].ptr, output_data);
    return TNN_OK;
}

REGISTER_CUDA_ACC(Permute, LAYER_PERMUTE);

}  // namespace TNN_NS