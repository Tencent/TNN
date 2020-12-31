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

DECLARE_CUDA_ACC(HardSwish, LAYER_HARDSWISH);

__global__ void hardSwish_kernel(int count, const float *src1, const float *src2, float *dst, float alpha, float beta) {
    CUDA_KERNEL_LOOP(index, count) {
        dst[index] = src1[index] * max(min(src2[index] * alpha + beta, 1.f), 0.f);
    }
}

Status CudaHardSwishLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaHardSwishLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaHardSwishLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<HardSwishLayerParam *>(param_);
    if (!params) {
        LOGE("Error: HardSwishLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: HardSwishLayerParam is nil");
    }

    int count = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims);
    float* input_data1 = static_cast<float*>(inputs[0]->GetHandle().base);
    float* output_data = static_cast<float*>(outputs[0]->GetHandle().base);
    float* input_data2 = input_data1;
    if (inputs.size() != 1) {
        input_data2 = static_cast<float*>(inputs[1]->GetHandle().base);
    }
    hardSwish_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
        count, input_data1, input_data2, output_data, params->alpha, params->beta);
    
    return TNN_OK;
}

REGISTER_CUDA_ACC(HardSwish, LAYER_HARDSWISH);

}  // namespace TNN_NS