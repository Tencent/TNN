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
// specific language governing permissions and limitations under the License./

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(PixelShuffle, LAYER_PIXEL_SHUFFLE);

__global__ void pixel_shuffle_kernel(int count, const float* input, float* output, int slice_size, int upscale_factor,
        int input_h, int input_w) {
    CUDA_KERNEL_LOOP(index, count) {
        int s = (index / upscale_factor / upscale_factor / input_h / input_w);
        int i = (index / upscale_factor / input_h / input_w) % upscale_factor;
        int j = (index / input_h / input_w) % upscale_factor;
        int h = (index / input_w) % input_h;
        int w = index % input_w;
        output[s * input_h * upscale_factor * input_w * upscale_factor +
            h * upscale_factor * input_w * upscale_factor + i * input_w * upscale_factor +
            w * upscale_factor + j] = input[s * upscale_factor * upscale_factor * input_h * input_w +
            i * upscale_factor * input_h * input_w + j * input_h * input_w + h * input_w + w];
    }
}

Status CudaPixelShuffleLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaPixelShuffleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaPixelShuffleLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param   = dynamic_cast<PixelShuffleLayerParam *>(param_);
    int upscale_factor = layer_param->upscale_factor;
    auto input_blob    = inputs[0];
    auto input_dims    = input_blob->GetBlobDesc().dims;
    auto output_blob   = outputs[0];
    auto output_dims   = output_blob->GetBlobDesc().dims;
    int slice_size     = DimsVectorUtils::Count(output_dims, 0, 2);
    auto input_h       = input_dims[2];
    auto input_w       = input_dims[3];
    auto count         = slice_size * upscale_factor * upscale_factor * input_h * input_w;
    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto input_prt  = static_cast<float *>(input_blob->GetHandle().base);
        auto output_ptr = static_cast<float *>(output_blob->GetHandle().base);
        pixel_shuffle_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            count, input_prt, output_ptr, slice_size, upscale_factor, input_h, input_w);
    }
    return TNN_OK;
}

REGISTER_CUDA_ACC(PixelShuffle, LAYER_PIXEL_SHUFFLE);

}  // namespace TNN_NS
