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

DECLARE_CUDA_ACC(HardSigmoid, LAYER_HARDSIGMOID);

 __global__ void hard_sigmoid_kernel(const int n, const float* in, float* out,
        const float alpha, const float beta) {
    const float minV  = -beta / alpha;
    const float maxV  = (1.0f - beta) / alpha;
    CUDA_KERNEL_LOOP(index, n) {
        float value = in[index];
        if (value <= minV) {
            value = 0.0;
        } else if (value < maxV) {
            value = value * alpha + beta;
        } else {
            value = 1.0;
        }
        out[index] = value;
    }
}

Status CudaHardSigmoidLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaHardSigmoidLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaHardSigmoidLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<HardSigmoidLayerParam *>(param_);
    if (!params) {
        LOGE("Error: hardsigmoid layer param is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: hardsigmoid layer param is nil");
    }
    const float alpha_ = params->alpha;
    const float beta_  = params->beta;

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    int count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    hard_sigmoid_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
        count, input_data, output_data, alpha_, beta_);
    return TNN_OK;
}

REGISTER_CUDA_ACC(HardSigmoid, LAYER_HARDSIGMOID);

}  // namespace TNN_NS