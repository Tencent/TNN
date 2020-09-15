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

DECLARE_CUDA_ACC(Clip, LAYER_CLIP);

__global__ void clip_kernel(const int n, const float* in, float* out, float min, float max) {
    CUDA_KERNEL_LOOP(index, n) {
        float value = in[index] > min ? in[index] : min;
        out[index] = value < max ? value : max;
    }
}

Status CudaClipLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaClipLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaClipLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<ClipLayerParam *>(param_);
    if (!params) {
        LOGE("Error: layer param is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    int count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    clip_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
        count, input_data, output_data, params->min, params->max);
    return TNN_OK;
}

REGISTER_CUDA_ACC(Clip, LAYER_CLIP);

}  // namespace TNN_NS