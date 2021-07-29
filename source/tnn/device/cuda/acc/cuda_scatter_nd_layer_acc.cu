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

DECLARE_CUDA_ACC(ScatterND, LAYER_SCATTER_ND);

__global__ void scatter_nd_kernel(int offset_count, const int* indice, const float* update, float* output,
        int last_indice_dimension, int element_to_copy, int* element_counts) {
    CUDA_KERNEL_LOOP(index, offset_count) {
        int offset = 0;
        for (int j = 0; j < last_indice_dimension; j++) {
            offset += indice[index * last_indice_dimension + j] * element_counts[j];
        }
        for (int j = 0; j < element_to_copy; j++) {
            output[offset] = update[index * element_to_copy + j];
        }
    }
}

Status CudaScatterNDLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    auto res = dynamic_cast<ScatterNDLayerResource *>(resource);
    if (ret != TNN_OK) {
        return ret;
    }

    if (inputs.size() < 3) {
        auto count = res->indices.GetDataCount();
        CreateTempBuf(count * sizeof(int));
    }

    return TNN_OK;
}

Status CudaScatterNDLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    this->is_reshaped = false;
    return TNN_OK;
}

Status CudaScatterNDLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *update_blob = inputs.size() < 3 ? inputs[1] : inputs[2];
    Blob *output_blob = outputs[0];
    auto input_dims = inputs[0]->GetBlobDesc().dims;
    float* input_data = reinterpret_cast<float*>(input_blob->GetHandle().base);
    float* update_data = reinterpret_cast<float*>(update_blob->GetHandle().base);
    float* output_data = reinterpret_cast<float*>(output_blob->GetHandle().base);
    int* indice_data = nullptr;
    int* element_counts = nullptr;

    if (inputs.size() < 3 && tempbufs_.size() < 2) {
        CreateTempBuf(input_dims.size() * sizeof(int));
    }

    auto resource = dynamic_cast<ScatterNDLayerResource *>(resource_);
    DimsVector indices_dims;

    if (!this->is_reshaped) {
        if (inputs.size() < 3) {
            indices_dims = resource->indices.GetBufferDims();
        } else {
            indices_dims = inputs[1]->GetBlobDesc().dims;
        }
        auto indice_rank = indices_dims.size();
        auto last_indice_dimension = indices_dims[indice_rank - 1];
        std::vector<int> element_counts_(last_indice_dimension, 0);
        for (int i = 0; i < last_indice_dimension; ++i) {
            element_counts_[i] = DimsVectorUtils::Count(input_dims, i + 1);
        }

        if (inputs.size() < 3) {
            int* indice = resource->indices.force_to<int*>();
            int count = resource->indices.GetDataCount();
            cudaMemcpyAsync(tempbufs_[0].ptr, indice, count * sizeof(int), cudaMemcpyHostToDevice, context_->GetStream());
            cudaMemcpyAsync(tempbufs_[1].ptr, element_counts_.data(), last_indice_dimension * sizeof(int),
                cudaMemcpyHostToDevice, context_->GetStream());
        } else {
            cudaMemcpyAsync(tempbufs_[0].ptr, element_counts_.data(), last_indice_dimension * sizeof(int),
                cudaMemcpyHostToDevice, context_->GetStream());
        }
        this->is_reshaped = true;
    }

    if (inputs.size() < 3) {
        indices_dims = resource->indices.GetBufferDims();
        indice_data = (int*)tempbufs_[0].ptr;
        element_counts = (int*)tempbufs_[1].ptr;
    } else {
        indices_dims = inputs[1]->GetBlobDesc().dims;
        indice_data = reinterpret_cast<int*>(inputs[1]->GetHandle().base);
        element_counts = (int*)tempbufs_[0].ptr;
    }

    auto indice_rank = indices_dims.size();
    auto last_indice_dimension = indices_dims[indice_rank - 1];
    int element_to_copy = DimsVectorUtils::Count(input_dims, last_indice_dimension);
    int offset_count = DimsVectorUtils::Count(indices_dims, 0, indice_rank - 1);

    cudaMemcpyAsync(output_data, input_data, DimsVectorUtils::Count(input_dims) * sizeof(float),
        cudaMemcpyDeviceToDevice, context_->GetStream());

    scatter_nd_kernel<<<TNN_CUDA_GET_BLOCKS(offset_count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
        offset_count, indice_data, update_data, output_data, last_indice_dimension, element_to_copy, element_counts);

    return TNN_OK;
}

REGISTER_CUDA_ACC(ScatterND, LAYER_SCATTER_ND);

}  // namespace TNN_NS
