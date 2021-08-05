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
        int axis, float* output_data) {
    int indices_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (indices_index >= indices_size) return;

    int data_idx = 0;
    int dim = indices_index;
    int remain = indices_index;
    for (int i = 0; i < rank; ++i) {
        dim = dim / remain;
        remain = remain - dim * indices_dims[i];
        if (i == axis) {
            dim = indices_data[indices_index];
            if (dim < -input_dims[i] || dim >= input_dims[i]) return;
            if (dim < 0) dim += input_dims[i];
        }
        int prod = 1;
        for (int j = i + 1; j < rank; ++j) prod *= input_dims[j];
        data_idx += prod * dim;
    }
    output_data[data_idx] = updates[indices_index];
}

Status CudaScatterElementsLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }

    if (inputs.size() < 3) {
        auto res = dynamic_cast<ScatterElementsLayerResource *>(resource);
        auto size = res->data.GetBytesSize();
        CreateTempBuf(size);
        void* data = res->data.force_to<void*>();
        CUDA_CHECK(cudaMemcpyAsync(tempbufs_[0].ptr, data, size, cudaMemcpyHostToDevice, context_->GetStream()));
    }

    return TNN_OK;
}

Status CudaScatterElementsLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaScatterElementsLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ScatterElementsLayerParam *>(param_);
    auto resource = dynamic_cast<ScatterElementsLayerResource *>(resource_);

    float* output_data = reinterpret_cast<float*>(outputs[0]->GetHandle().base);

    float* data_ptr;
    DimsVector data_dims;
    int* indices_data;
    DimsVector indices_dims;
    float* update_data;
    DimsVector update_dims;
    int data_size;
    int tempbuf_index;

    if (inputs.size() < 3) {
        data_ptr = (float*)tempbufs_[0].ptr;
        data_dims = resource->data.GetBufferDims();
        indices_data = (int*)inputs[0]->GetHandle().base;
        indices_dims = inputs[0]->GetBlobDesc().dims;
        update_data = (float*)inputs[1]->GetHandle().base;
        update_dims = inputs[1]->GetBlobDesc().dims;
        data_size = resource->data.GetBytesSize();
        tempbuf_index = 1;
    } else {
        data_ptr = (float*)inputs[0]->GetHandle().base;
        data_dims = inputs[0]->GetBlobDesc().dims;
        indices_data = (int*)inputs[1]->GetHandle().base;
        indices_dims = inputs[1]->GetBlobDesc().dims;
        update_data = (float*)inputs[2]->GetHandle().base;
        update_dims = inputs[2]->GetBlobDesc().dims;
        data_size = DimsVectorUtils::Count(data_dims);
        tempbuf_index = 0;
    }

    auto data_rank = data_dims.size();
    int indices_size = DimsVectorUtils::Count(indices_dims);
    int axis = param->axis < 0 ? param->axis + data_rank : param->axis;

    CUDA_CHECK(cudaMemcpyAsync(output_data, data_ptr, data_size, cudaMemcpyDeviceToDevice, context_->GetStream()));

    if (!this->is_reshaped) {
        CreateTempBuf(data_rank * sizeof(int));
        CreateTempBuf(data_rank * sizeof(int));

        CUDA_CHECK(cudaMemcpyAsync(tempbufs_[tempbuf_index].ptr, data_dims.data(), data_rank * sizeof(int),
            cudaMemcpyHostToDevice, context_->GetStream()));
        CUDA_CHECK(cudaMemcpyAsync(tempbufs_[tempbuf_index+1].ptr, indices_dims.data(), data_rank * sizeof(int),
            cudaMemcpyHostToDevice, context_->GetStream()));
        this->is_reshaped = true;
    }

    int grid = (indices_size + TNN_CUDA_NUM_THREADS - 1) / TNN_CUDA_NUM_THREADS;
    scatter_elements_kernel<<<grid, TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(data_rank, data_ptr,
        (int*)tempbufs_[tempbuf_index].ptr, indices_data, indices_size, (int*)tempbufs_[tempbuf_index+1].ptr,
        update_data, axis, output_data);

    return TNN_OK;
}

REGISTER_CUDA_ACC(ScatterElements, LAYER_SCATTER_ELEMENTS);

}  // namespace TNN_NS
