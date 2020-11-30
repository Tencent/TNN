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

DECLARE_CUDA_ACC(Pooling, LAYER_POOLING);

__global__ void pool_kernel(int wh, float *in, float* out, int n, int c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * c) return;
    float sum = 0.f;
    for (int i = 0; i < wh; i++) {
        sum += in[tid * wh + i];
    }
    out[tid] = sum / wh;
}

Status CudaPoolingLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaPoolingLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaPoolingLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto dims = input_blob->GetBlobDesc().dims;
    int wh = dims[2] * dims[3];
    int c = dims[1];
    int n = dims[0];
    int grid = (n * c + 127) / 128;
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    pool_kernel<<<grid, 128, 0, context_->GetStream()>>>(wh, input_data, output_data, n, c);
    return TNN_OK;
}

REGISTER_CUDA_ACC(Pooling, LAYER_POOLING);

}  // namespace TNN_NS
