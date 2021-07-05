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

DECLARE_CUDA_ACC(PReLU, LAYER_PRELU);

__global__ void prelu_kernel(const int n, const int channels, const int dim,
        const float* in, float* out, const float* slope_data, const int div_factor) {
    CUDA_KERNEL_LOOP(index, n) {
        int c = (index / dim) % channels / div_factor;
        out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
    }
}

Status CudaPReLULayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }

    auto res = dynamic_cast<PReluLayerResource *>(resource_);
    if (!res) {
        LOGE("Error: PReluLayerResource is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: PReluLayerResource is nil");
    }
    const int slope_size = res->slope_handle.GetBytesSize();
    const float *slope_data = res->slope_handle.force_to<float *>();
    CreateTempBuf(slope_size);
    cudaMemcpyAsync(tempbufs_[0].ptr, slope_data, slope_size, cudaMemcpyHostToDevice, context_->GetStream());

    return TNN_OK;
}

Status CudaPReLULayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaPReLULayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<PReluLayerParam *>(param_);
    if (!params) {
        LOGE("Error: PReluLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: PReluLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    const int channels = output_blob->GetBlobDesc().dims[1];
    const int count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    const int hw = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);
    if (0 == hw) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }
    const int div_factor = params->channel_shared ? channels : 1;
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    prelu_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>
        (count, channels, hw, input_data, output_data, (const float*)tempbufs_[0].ptr, div_factor);    
    return TNN_OK;
}

REGISTER_CUDA_ACC(PReLU, LAYER_PRELU);

}  // namespace TNN_NS