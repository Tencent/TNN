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

DECLARE_CUDA_ACC(Tile, LAYER_REPEAT);

__global__ void tile_kernel(int count, const float *input, float *output, const int* input_dims, const int* output_dims, int size) {
    CUDA_KERNEL_LOOP(index, count) {
        int offset = 0;
        int prod = count;
        for (int i = 0; i < size; i++) {
            prod /= output_dims[i];
            int mod = index / prod % input_dims[i];
            offset = offset * input_dims[i] + mod;
        }
        output[index] = input[offset];
    }
}

Status CudaTileLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaTileLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    this->is_reshaped = false;
    return TNN_OK;
}

Status CudaTileLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (tempbufs_.size() == 0) {
        auto output_dims = outputs[0]->GetBlobDesc().dims;
        CreateTempBuf(output_dims.size() * sizeof(int));
        CreateTempBuf(output_dims.size() * sizeof(int));
    }

    if (!this->is_reshaped) {
        auto input_dims = inputs[0]->GetBlobDesc().dims;
        auto output_dims = outputs[0]->GetBlobDesc().dims;
        while (input_dims.size() < output_dims.size()) {
            input_dims.insert(input_dims.begin(), 1);
        }

        cudaMemcpyAsync(tempbufs_[0].ptr, input_dims.data(), input_dims.size()*sizeof(int),
            cudaMemcpyHostToDevice, context_->GetStream());
        cudaMemcpyAsync(tempbufs_[1].ptr, output_dims.data(), output_dims.size()*sizeof(int),
            cudaMemcpyHostToDevice, context_->GetStream());
        this->is_reshaped = true;
    }

    int count = DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims);
    float* input_data = static_cast<float*>(inputs[0]->GetHandle().base);
    float* output_data = static_cast<float*>(outputs[0]->GetHandle().base);

    tile_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
        count, input_data, output_data, (const int *)tempbufs_[0].ptr, (const int *)tempbufs_[1].ptr,
        inputs[0]->GetBlobDesc().dims.size());
    return TNN_OK;
}

REGISTER_CUDA_ACC(Tile, LAYER_REPEAT);

}  // namespace TNN_NS
