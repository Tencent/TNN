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

DECLARE_CUDA_ACC(Upsample, LAYER_UPSAMPLE);

__global__ void upsample_nearest2d_kernel(int count, const float* srcData,
        float* dstData, float rheight, float rwidth, int output_c,
        int output_h, int output_w, int input_h, int input_w) {
    CUDA_KERNEL_LOOP(index, count) {
        int w = index % output_w;
        int h = (index / output_w) % output_h;
        int nc = index / output_w / output_h;
        int scaled_h = h * rheight;
        int scaled_w = w * rwidth;
        int in_index = (nc * input_h + scaled_h) * input_w + scaled_w;
        dstData[index] = srcData[in_index];
    }
}

__global__ void upsample_bilinear2d_align_corners_kernel(int count, const float * srcData,
        float * dstData, float rheight, float rwidth, int output_c,
        int output_h, int output_w, int input_h, int input_w) {
    CUDA_KERNEL_LOOP(index, count) {
        int w = index % output_w;
        int h = (index / output_w) % output_h;
        int c = (index / output_w / output_h) % output_c;
        int n = index / output_w / output_h / output_c;
        float h1r = rheight * h;
        int h1 = h1r;
        int h1p = (h1 < input_h - 1) ? 1 : 0;
        float h1lambda = h1r - h1;
        float h0lambda = (float)1. - h1lambda;

        float w1r = rwidth * w;
        int w1 = w1r;
        int w1p = (w1 < input_w - 1) ? 1 : 0;
        float w1lambda = w1r - w1;
        float w0lambda = (float)1. - w1lambda;

        int left_top = (n * output_c + c) * input_h * input_w + h1 * input_w + w1;
        dstData[index] = h0lambda * (w0lambda * srcData[left_top] + w1lambda * srcData[left_top + w1p]) +
            h1lambda * (w0lambda * srcData[left_top + h1p * input_w] + w1lambda * srcData[left_top + h1p * input_w + w1p]);
    }
}

__global__ void upsample_bilinear2d_no_align_corners_kernel(int count, const float * srcData,
        float * dstData, float rheight, float rwidth, int output_c,
        int output_h, int output_w, int input_h, int input_w) {
    CUDA_KERNEL_LOOP(index, count) {
        int w = index % output_w;
        int h = (index / output_w) % output_h;
        int c = (index / output_w / output_h) % output_c;
        int n = index / output_w / output_h / output_c;
        float h1r = rheight * (h + 0.5) - 0.5;
        h1r = h1r >= 0 ? h1r : 0;
        int h1 = h1r;
        int h1p = (h1 < input_h - 1) ? 1 : 0;
        float h1lambda = h1r - h1;
        float h0lambda = (float)1. - h1lambda;

        float w1r = rwidth * (w + 0.5) - 0.5;
        w1r = w1r >= 0? w1r : 0;
        int w1 = w1r;
        int w1p = (w1 < input_w - 1) ? 1 : 0;
        float w1lambda = w1r - w1;
        float w0lambda = (float)1. - w1lambda;

        int left_top = (n * output_c + c) * input_h * input_w + h1 * input_w + w1;
        dstData[index] = h0lambda * (w0lambda * srcData[left_top] + w1lambda * srcData[left_top + w1p]) +
            h1lambda * (w0lambda * srcData[left_top + h1p * input_w] + w1lambda * srcData[left_top + h1p * input_w + w1p]);
    }
}

Status CudaUpsampleLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaUpsampleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaUpsampleLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<UpsampleLayerParam *>(param_);
    if (!params) {
        LOGE("Error: UpsampleLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: UpsampleLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    int input_height = input_blob->GetBlobDesc().dims[2];
    int input_width = input_blob->GetBlobDesc().dims[3];
    int output_height = output_blob->GetBlobDesc().dims[2];
    int output_width = output_blob->GetBlobDesc().dims[3];
    int output_channel = output_blob->GetBlobDesc().dims[1];
    int count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    float* input_data = static_cast<float*>(input_blob->GetHandle().base);
    float* output_data = static_cast<float*>(output_blob->GetHandle().base);
    if (params->mode == 2) {
        if (params->align_corners) {
            float rheight = (output_height > 1) ? (float)(input_height - 1) / (output_height - 1) : 0.f;
            float rwidth = (output_width > 1) ? (float)(input_width - 1) / (output_width - 1) : 0.f;
            upsample_bilinear2d_align_corners_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(count,
                input_data, output_data, rheight, rwidth, output_channel, output_height, output_width, input_height, input_width);
        } else {
            float rheight = (output_height > 1) ? (float)(input_height) / output_height : 0.f;
            float rwidth = (output_width > 1) ? (float)(input_width) / output_width : 0.f;
            upsample_bilinear2d_no_align_corners_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(count,
                input_data, output_data, rheight, rwidth, output_channel, output_height, output_width, input_height, input_width);
        }
    } else if (params->mode == 1) {
        float rheight = (float)(input_height) / output_height;
        float rwidth = (float)(input_width) / output_width;
        upsample_nearest2d_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(count,
            input_data, output_data, rheight, rwidth, output_channel, output_height, output_width, input_height, input_width);
    } else {
        LOGE("Error: Upsample dont support resize mode\n");
        return Status(TNNERR_MODEL_ERR, "Error: Upsample dont support resize mode");
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(Upsample, LAYER_UPSAMPLE);

}  // namespace TNN_NS