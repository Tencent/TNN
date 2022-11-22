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

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype *a, const Dtype *b, Dtype *y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = a[index] * b[index];
    }
}

DECLARE_CUDA_ACC(Mul, LAYER_MUL);

Status CudaMulLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                             const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaMulLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaMulLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input0 = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];
    auto dtype  = input0->GetBlobDesc().data_type;

    auto input0_data = reinterpret_cast<void *>(input0->GetHandle().base);
    auto input1_data = reinterpret_cast<void *>(input1->GetHandle().base);
    auto output_data = reinterpret_cast<void *>(output->GetHandle().base);

    auto output_dims = output->GetBlobDesc().dims;
    auto count       = DimsVectorUtils::Count(output_dims);

    if (dtype == DATA_TYPE_FLOAT) {
        mul_kernel<float><<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            count, (const float *)input0_data, (const float *)input1_data, (float *)output_data);
    } else if (dtype == DATA_TYPE_HALF) {
        mul_kernel<__half><<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            count, (const __half *)input0_data, (const __half *)input1_data, (__half *)output_data);
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(Mul, LAYER_MUL);

}  // namespace TNN_NS