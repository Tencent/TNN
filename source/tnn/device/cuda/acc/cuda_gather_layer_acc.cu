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
#include "tnn/device/cuda/acc/cuda_gather_layer_acc.h"
#include "tnn/device/cuda/acc/cuda_gather_layer_acc_kernel.cuh"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status CudaGatherLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(CudaLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    auto layer_param = dynamic_cast<GatherLayerParam *>(param);
    auto layer_resource = dynamic_cast<GatherLayerResource *>(resource);

    if ((layer_param->data_in_resource || layer_param->indices_in_resource) && !layer_resource) {
        return Status(TNNERR_MODEL_ERR, "Gather resource is invalid");
    }

    if (layer_param->data_in_resource) {
        auto input_data = layer_resource->data.force_to<char*>();
        auto input_size = layer_resource->data.GetBytesSize();
        CUDA_CHECK(cudaMalloc((void **)&input_data_, input_size));
        CUDA_CHECK(cudaMemcpy(input_data_, input_data, input_size, cudaMemcpyHostToDevice));
    }

    if (layer_param->indices_in_resource) {
        auto indices_data = layer_resource->indices.force_to<char*>();
        auto indices_size = layer_resource->indices.GetBytesSize();
        CUDA_CHECK(cudaMalloc((void **)&indices_data_, indices_size));
        CUDA_CHECK(cudaMemcpy(indices_data_, indices_data, indices_size, cudaMemcpyHostToDevice));
    }
    return TNN_OK;
}

CudaGatherLayerAcc::~CudaGatherLayerAcc(){
    if (input_data_ != nullptr) {
        CUDA_CHECK(cudaFree(input_data_));
    }
    if (indices_data_ != nullptr) {
        CUDA_CHECK(cudaFree(indices_data_));
    }
}

Status CudaGatherLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaGatherLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<GatherLayerParam *>(param_);
    auto layer_resource = dynamic_cast<GatherLayerResource *>(resource_);

    DataType dtype;
    DimsVector input_dims;
    void *input_data_ptr = nullptr;
    if (layer_param->data_in_resource) {
        input_dims = layer_resource->data.GetBufferDims();
        input_data_ptr = input_data_;
        dtype = layer_resource->data.GetDataType();
    } else {
        input_dims = (*(inputs.begin()))->GetBlobDesc().dims;
        input_data_ptr = (*(inputs.begin()))->GetHandle().base;
        dtype = (*(inputs.begin()))->GetBlobDesc().data_type;
    }

    DimsVector indices_dims;
    void *indices_data_ptr = nullptr;
    if (layer_param->indices_in_resource) {
        indices_dims = layer_resource->indices.GetBufferDims();
        indices_data_ptr = indices_data_;
    } else {
        indices_dims = (*(inputs.rbegin()))->GetBlobDesc().dims;
        indices_data_ptr = (*(inputs.rbegin()))->GetHandle().base;
    }

    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;
    void* output_data_ptr = outputs[0]->GetHandle().base;

    int axis = layer_param->axis;
    int dst_size = DimsVectorUtils::Count(output_dims);
    int slice_size = DimsVectorUtils::Count(input_dims, axis + 1);
    int input_slice_count = DimsVectorUtils::Count(input_dims, axis, axis + 1);
    int output_slice_count = DimsVectorUtils::Count(indices_dims);

    if (dtype == DATA_TYPE_FLOAT) {
        return RunGather(dst_size, slice_size, input_slice_count, output_slice_count,
                         (const float*)input_data_ptr, (const int*)indices_data_ptr,
                         (float*)output_data_ptr, context_->GetStream());
    } else if (dtype == DATA_TYPE_INT32) {
        return RunGather(dst_size, slice_size, input_slice_count, output_slice_count,
                         (const int*)input_data_ptr, (const int*)indices_data_ptr,
                         (int*)output_data_ptr, context_->GetStream());
    } else if (dtype == DATA_TYPE_HALF) {
        return RunGather(dst_size, slice_size, input_slice_count, output_slice_count,
                         (const __half*)input_data_ptr, (const int*)indices_data_ptr,
                         (__half*)output_data_ptr, context_->GetStream());
    } else {
        LOGE("CudaGatherLayerAcc::Forward unsupported dtype %d\n", dtype);
        return Status(TNNERR_MODEL_ERR, "CudaGatherLayerAcc::Forward unsupported dtype");
    }
}

REGISTER_CUDA_ACC(Gather, LAYER_GATHER);

}  // namespace TNN_NS