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
// COElementsITIONS OF ANY KIElements, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(ScatterElements, LAYER_SCATTER_ELEMENTS);

__global__ void scatter_elements_kernel(int rank, const float* input_data, int* input_dims,
        const int* indices_data, int indices_size, int* indices_dims, const float* updates,
        int axis, int op, float* output_data) {
    int indices_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (indices_index >= indices_size) return;

    int data_idx = 0;
    int dim, remain = indices_index;
    int indices_strides = indices_size;
    for (int i = 0; i < rank; ++i) {
        indices_strides /= indices_dims[i];
        dim = remain / indices_strides;
        remain = remain - dim * indices_strides;
        if (i == axis) {
            dim = indices_data[indices_index];
            if (dim < -input_dims[i] || dim >= input_dims[i]) return;
            if (dim < 0) dim += input_dims[i];
        }
        int prod = 1;
        for (int j = i + 1; j < rank; ++j) prod *= input_dims[j];
        data_idx += prod * dim;
    }
    if (op == 0) {
        output_data[data_idx] = updates[indices_index];
    } else {
        atomicAdd(&output_data[data_idx], updates[indices_index]);
    }
}

Status CudaScatterElementsLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaScatterElementsLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaScatterElementsLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ScatterElementsLayerParam *>(param_);
    float* output_data = reinterpret_cast<float*>(outputs[0]->GetHandle().base);

    float* data_ptr = (float*)inputs[0]->GetHandle().base;
    DimsVector data_dims = inputs[0]->GetBlobDesc().dims;
    int* indices_data = (int*)inputs[1]->GetHandle().base;
    DimsVector indices_dims = inputs[1]->GetBlobDesc().dims;
    float* update_data = (float*)inputs[2]->GetHandle().base;
    DimsVector update_dims = inputs[2]->GetBlobDesc().dims;
    int data_size = DimsVectorUtils::Count(data_dims);

    auto data_rank = data_dims.size();
    int indices_size = DimsVectorUtils::Count(indices_dims);
    int axis = param->axis < 0 ? param->axis + data_rank : param->axis;

    CUDA_CHECK(cudaMemcpyAsync(output_data, data_ptr, data_size * sizeof(float),
        cudaMemcpyDeviceToDevice, context_->GetStream()));

    if (!this->is_reshaped) {
        CreateTempBuf(data_rank * sizeof(int));
        CreateTempBuf(data_rank * sizeof(int));

        CUDA_CHECK(cudaMemcpyAsync(tempbufs_[0].ptr, data_dims.data(), data_rank * sizeof(int),
            cudaMemcpyHostToDevice, context_->GetStream()));
        CUDA_CHECK(cudaMemcpyAsync(tempbufs_[1].ptr, indices_dims.data(), data_rank * sizeof(int),
            cudaMemcpyHostToDevice, context_->GetStream()));
        this->is_reshaped = true;
    }

    int grid = (indices_size + TNN_CUDA_NUM_THREADS - 1) / TNN_CUDA_NUM_THREADS;
    scatter_elements_kernel<<<grid, TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(data_rank, data_ptr,
        (int*)tempbufs_[0].ptr, indices_data, indices_size, (int*)tempbufs_[1].ptr,
        update_data, axis, param->op, output_data);

    return TNN_OK;
}

REGISTER_CUDA_ACC(ScatterElements, LAYER_SCATTER_ELEMENTS);

}  // namespace TNN_NS
