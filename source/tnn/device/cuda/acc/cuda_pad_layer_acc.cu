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

DECLARE_CUDA_ACC(Pad, LAYER_PAD);

__global__ void pad_default_kernel(const float* src, float* dst, int count, int input_channel, int output_channel,
        int pad_c, int output_h, int output_w, int input_h, int input_w, int pad_w, int pad_h, float value) {
    CUDA_KERNEL_LOOP(idx, count) {
        int dst_n = idx / (output_channel * output_h * output_w);
        int dst_c = (idx / (output_h * output_w)) % output_channel;
        int dst_h = (idx / output_w) % output_h;
        int dst_w = idx % output_w;

        if (dst_c < pad_c || dst_c >= input_channel + pad_c || dst_h < pad_h || dst_h >= (pad_h + input_h) ||
                dst_w < pad_w || dst_w >= (pad_w + input_w)) {
            dst[idx] = value;
        } else {
          int src_idx = dst_n * input_channel * input_h * input_w + (dst_c - pad_c) * input_h * input_w +
            (dst_h - pad_h) * input_w + (dst_w - pad_w);
          dst[idx] = src[src_idx];
        }
    }
}

__global__ void pad_symmetric_kernel(const float* src, float* dst, int count, int channels, int output_h,
        int output_w, int input_h, int input_w, int pad_w, int pad_h) {
    CUDA_KERNEL_LOOP(idx, count) {
        int dst_n = idx / (channels * output_h * output_w);
        int dst_c = (idx / (output_h * output_w)) % channels;
        int dst_h = (idx / output_w) % output_h;
        int dst_w = idx % output_w;

        int h = dst_h >= pad_h? (dst_h < pad_h + input_h? dst_h - pad_h : pad_h - 1 - dst_h + 2 * input_h) : pad_h - 1 - dst_h;
        int w = dst_w >= pad_w? (dst_w < pad_w + input_w? dst_w - pad_w : pad_w - 1 - dst_w + 2 * input_w) : pad_w - 1 - dst_w;
        dst[idx] = src[dst_n * channels * input_h * input_w + dst_c * input_h * input_w + h * input_w + w];
    }
}

__global__ void pad_reflect_kernel(const float* src, float* dst, int count, int channels, int output_h, int output_w,
        int input_h, int input_w, int pad_w, int pad_h) {
    CUDA_KERNEL_LOOP(idx, count) {
        int dst_n = idx / (channels * output_h * output_w);
        int dst_c = (idx / (output_h * output_w)) % channels;
        int dst_h = (idx / output_w) % output_h;
        int dst_w = idx % output_w;

        int h = dst_h >= pad_h? (dst_h < pad_h + input_h? dst_h - pad_h : pad_h - 2 - dst_h + 2 * input_h) : pad_h - dst_h;
        int w = dst_w >= pad_w? (dst_w < pad_w + input_w? dst_w - pad_w : pad_w - 2 - dst_w + 2 * input_w) : pad_w - dst_w;
        dst[idx] = src[dst_n * channels * input_h * input_w + dst_c * input_h * input_w + h * input_w + w];
    }
}

__global__ void pad_reflect_kernel_fp16(const __half * src, __half * dst, int count, int channels, int output_h, int output_w,
        int input_h, int input_w, int pad_w, int pad_h) {
    const float4 * src_l = reinterpret_cast<const float4*>(src);
    float4 * dst_l = reinterpret_cast<float4*>(dst);

    channels = channels / 8;
    CUDA_KERNEL_LOOP(idx, count / 8) {
        int dst_n = idx / (channels * output_h * output_w);
        int dst_h = (idx / (output_w * channels)) % output_h;
        int dst_w = (idx / channels) % output_w;
        int dst_c = idx % channels;

        int h = dst_h >= pad_h ? (dst_h < pad_h + input_h ? dst_h - pad_h : pad_h - 2 - dst_h + 2 * input_h) : pad_h - dst_h;
        int w = dst_w >= pad_w ? (dst_w < pad_w + input_w ? dst_w - pad_w : pad_w - 2 - dst_w + 2 * input_w) : pad_w - dst_w;

        int src_idx = dst_n * input_h * input_w * channels + h * input_w * channels + w * channels + dst_c;

        dst_l[idx] = src_l[src_idx];
    }
}

Status CudaPadLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaPadLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaPadLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<PadLayerParam *>(param_);
    if (!params) {
        LOGE("Error: PadLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: PadLayerParam is nil");
    }

    int pad_l = params->pads[0];
    int pad_r = params->pads[1];
    int pad_t = params->pads[2];
    int pad_b = params->pads[3];
    int pad_c_b = params->pads[4];
    int pad_c_e = params->pads[5];
    float value = params->value;

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    const int count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    int output_channel = output_blob->GetBlobDesc().dims[1];
    int input_h = input_blob->GetBlobDesc().dims[2];
    int input_w = input_blob->GetBlobDesc().dims[3];
    int output_h = output_blob->GetBlobDesc().dims[2];
    int output_w = output_blob->GetBlobDesc().dims[3];
    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float* input_data = static_cast<float*>(input_blob->GetHandle().base);
        float* output_data = static_cast<float*>(output_blob->GetHandle().base);

        if (params->type == 2) {
            pad_symmetric_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
                input_data, output_data, count, output_channel, output_h, output_w, input_h, input_w, pad_l,
                pad_t);
        } else if (params->type == 1) {
            pad_reflect_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
                input_data, output_data, count, output_channel, output_h, output_w, input_h, input_w, pad_l,
                pad_t);
        } else {
            int input_channel = input_blob->GetBlobDesc().dims[1];
            pad_default_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
                input_data, output_data, count, input_channel, output_channel, pad_c_b, output_h, output_w,
                input_h, input_w, pad_l, pad_t, value);
        }
    } else if (input_blob->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        if (params->type != 1) {
            LOGE("Error: Pad types other than relfect_mode do not supports half mode\n");
            return Status(TNNERR_MODEL_ERR, "Error: Pad layer don't supports pad type");
        }
        __half* input_data = static_cast<__half*>(input_blob->GetHandle().base);
        __half* output_data = static_cast<__half*>(output_blob->GetHandle().base);

        pad_reflect_kernel_fp16<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
            input_data, output_data, count, output_channel, output_h, output_w, input_h, input_w, pad_l,
            pad_t);
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc don't support datatype");
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(Pad, LAYER_PAD);

}  // namespace TNN_NS
