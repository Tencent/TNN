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

#include <cub/block/block_reduce.cuh>

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(Normalize, LAYER_NORMALIZE);

template<int p>
__device__ static void reduce_op(float &norm, const float &a) {
    norm = a;
}

template<> __device__ void reduce_op<1>(float& norm, const float &a) { norm += fabs(a); }
template<> __device__ void reduce_op<2>(float& norm, const float &a) { norm += a * a; }
template<> __device__ void reduce_op<INT_MAX>(float& norm, const float &a) { norm = max(norm, a); }
template<> __device__ void reduce_op<INT_MIN>(float& norm, const float &a) { norm = min(norm, a); }

template<int p>
__global__ void normalize_kernel(const float *__restrict src, float* dst, float eps, int num, int channel, int hw) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= hw) return;

    src += tid + blockIdx.y * channel * hw;
    dst += tid + blockIdx.y * channel * hw;
    float sum = 0.f;
    if (p == INT_MAX) {
        sum = -FLT_MAX;
    } else if (p == INT_MIN) {
        sum = FLT_MAX;
    }

    for (int c = 0; c < channel; c++) {
        reduce_op<p>(sum, src[c * hw]);
    }

    if (p == 2) {
        sum = max((float)sqrt(sum), eps);
    }

    for (int c = 0; c < channel; c++) {
        dst[c * hw] = src[c * hw] / sum;
    }
}

using kernel_function_ptr_t = decltype(&normalize_kernel<1>);

Status CudaNormalizeLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaNormalizeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaNormalizeLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<NormalizeLayerParam *>(param_);
    if (!params) {
        LOGE("Error: NormalizeLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: NormalizeLayerParam is nil");
    }

    float epsilon = params->epsilon;
    int axis = params->axis;
    int p = params->p;

    int across_spatial = params->across_spatial;

    // old tnn support scale the result of normalize and only norm2
    if ((p != 1 && p != 2 && p != INT_MAX && p != INT_MIN) || axis != 1 || across_spatial != 0) {
        LOGE("Error: layer param is not supported now\n");
        return Status(TNNERR_INST_ERR, "Error: layer param is not supported now");
    }

    kernel_function_ptr_t kernel_ptr = normalize_kernel<1>;
    switch (p) {
        case 1:
            kernel_ptr = normalize_kernel<1>;
            break;
        case 2:
            kernel_ptr = normalize_kernel<2>;
            break;
        case INT_MAX:
            kernel_ptr = normalize_kernel<INT_MAX>;
            break;
        case INT_MIN:
            kernel_ptr = normalize_kernel<INT_MIN>;
            break;
        default:
            kernel_ptr = normalize_kernel<1>;
            break;
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    auto output_dims = output_blob->GetBlobDesc().dims;
    int batch = output_dims[0];
    int channel = output_dims[1];
    int channel_size = DimsVectorUtils::Count(output_dims, 2);
    dim3 grid;
    const int block = 64;
    grid.x = (channel_size + block - 1) / block;
    grid.y = batch;
    kernel_ptr<<<grid, 64, 0, context_->GetStream()>>>(
        input_data, output_data, epsilon, batch, channel, channel_size);
    return TNN_OK;
}

REGISTER_CUDA_ACC(Normalize, LAYER_NORMALIZE);

}  // namespace TNN_NS