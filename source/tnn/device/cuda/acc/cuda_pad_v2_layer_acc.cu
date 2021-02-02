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

DECLARE_CUDA_ACC(PadV2, LAYER_PADV2);

__global__ void pad_default_kernel_v2(const float* src, float* dst, int count, int input_channel, int output_channel,
        int pad_c, int output_d, int output_h, int output_w, int input_d, int input_h, int input_w, int pad_d, int pad_h, int pad_w, float value) {
    CUDA_KERNEL_LOOP(idx, count) {
        int dst_n = idx / (output_channel * output_d * output_h * output_w);
        int dst_c = (idx / (output_d * output_h * output_w)) % output_channel;
        int dst_d = (idx / (output_h * output_w)) % output_d;
        int dst_h = (idx / output_w) % output_h;
        int dst_w = idx % output_w;

        if (dst_c < pad_c || dst_c >= input_channel + pad_c || dst_d < pad_d || dst_d >= input_d + pad_d || dst_h < pad_h || dst_h >= (pad_h + input_h) ||
                dst_w < pad_w || dst_w >= (pad_w + input_w)) {
            dst[idx] = value;
        } else {
          int src_idx = dst_n * input_channel * input_d * input_h * input_w + (dst_c - pad_c) * input_d * input_h * input_w +
            (dst_d - pad_d) * input_h * input_w + (dst_h - pad_h) * input_w + (dst_w - pad_w);
            dst[idx] = src[src_idx];
        }
    }
}

Status CudaPadV2LayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaPadV2LayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaPadV2LayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto params = dynamic_cast<PadLayerParam *>(param_);
    if (!params) {
        LOGE("Error: PadV2LayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: PadV2LayerParam is nil");
    }

    Blob* output_blob = outputs[0];
    Blob* input_blob = inputs[0];
    auto output_dims = output_blob->GetBlobDesc().dims;
    auto input_dims = input_blob->GetBlobDesc().dims;

    int pad_c = 0, pad_d = 0, pad_h = 0, pad_w = 0;
    int output_c = 1, output_d = 1, output_h = 1, output_w = 1;
    int input_c = 1, input_d = 1, input_h = 1, input_w = 1;

    pad_c = params->pads[1];  
    output_c = output_dims[1];
    input_c = input_dims[1];
    
    if(output_dims.size() > 2) {
        pad_d = params->pads[2];  
        output_d = output_dims[2];
        input_d = input_dims[2];
    }

    if(output_dims.size() > 3) {
        pad_h = params->pads[3];  
        output_h = output_dims[3];
        input_h = input_dims[3];
    }

    if(output_dims.size() > 4) {
        pad_w = params->pads[4];  
        output_w = output_dims[4];
        input_w = input_dims[4];
    }

    float value = params->value;

    const int count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float* input_data = static_cast<float*>(input_blob->GetHandle().base);
        float* output_data = static_cast<float*>(output_blob->GetHandle().base);

        if (params->type == 0) {
            pad_default_kernel_v2<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(
                input_data, output_data, count, input_c, output_c, pad_c, output_d, output_h, output_w,
                input_d, input_h, input_w, pad_d, pad_h, pad_w, value);
        } else {
            LOGE("Error: layer acc dont support pad type: %d\n", params->type);
            return Status(TNNERR_MODEL_ERR, "Error: layer acc don't support pad type");
        }
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc don't support datatype");
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(PadV2, LAYER_PADV2);

}  // namespace TNN_NS
