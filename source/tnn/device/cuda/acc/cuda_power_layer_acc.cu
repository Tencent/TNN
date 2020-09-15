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

DECLARE_CUDA_ACC(Pow, LAYER_POWER);

__global__ void pow_kernel(int n, const float* srcData, const float power, 
        const float scale, const float shift, float* dstData) {
   CUDA_KERNEL_LOOP(index, n) {
       float result = 1;
       float input = srcData[index] * scale + shift;
       for(int i = 0; i < power; ++i) {
           result *= input;
       }
       dstData[index] = result;
   } 
}

Status CudaPowLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaPowLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaPowLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<PowLayerParam *>(param_);
    if (!params) {
        LOGE("Error: PowLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: PowLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    int count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);

    pow_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
        count, input_data, params->exponent, params->scale, params->shift, output_data);
    return TNN_OK;
}

REGISTER_CUDA_ACC(Pow, LAYER_POWER);

}  // namespace TNN_NS