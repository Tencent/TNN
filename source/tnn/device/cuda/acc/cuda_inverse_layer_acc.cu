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

DECLARE_CUDA_ACC(Inverse, LAYER_INVERSE);

__global__ void inverse_kernel_2x2(int count, const float *input, float* output) {
    CUDA_KERNEL_LOOP(index, count) {
        const float* inptr = input + 4*index;
        float* outptr = output + 4*index;
        float det = inptr[0]*inptr[3] - inptr[1]*inptr[2];
        float det_inverse = 1.f / det;

        outptr[0] = inptr[3] * det_inverse;
        outptr[1] = -inptr[1] * det_inverse;
        outptr[2] = -inptr[2] * det_inverse;
        outptr[3] = inptr[0] * det_inverse;
    }
}

Status CudaInverseLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaInverseLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaInverseLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_dims = inputs[0]->GetBlobDesc().dims;
    float* input_data = static_cast<float*>(inputs[0]->GetHandle().base);
    float* output_data = static_cast<float*>(outputs[0]->GetHandle().base);
    int count = DimsVectorUtils::Count(input_dims, 0, (int)input_dims.size()-2);
    if (input_dims[input_dims.size()-1] == 2 && input_dims[input_dims.size()-2] == 2) {
        inverse_kernel_2x2<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            count, input_data, output_data);
    } else {
        return Status(TNNERR_PARAM_ERR, "CudaInverseLayerAcc now only support inverse of matrix batchx2x2");
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(Inverse, LAYER_INVERSE);

}  // namespace TNN_NS
