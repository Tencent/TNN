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

#include "tnn/utils/naive_compute.h"
#include "tnn/device/cpu/acc/cpu_layer_acc.h"

namespace TNN_NS {

DECLARE_CPU_ACC_WITH_FP32_RESOURCE(HdrGuide, LAYER_HDRGUIDE);

Status CpuHdrGuideLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuHdrGuideLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto resource = dynamic_cast<HdrGuideLayerResource *>(resource_);
    if (!resource) {
        return Status(TNNERR_MODEL_ERR, "Error: HdrGuideLayerResource is empty");
    }

    Blob *input_blob         = inputs[0];
    Blob *output_blob        = outputs[0];
    float *input_ptr         = static_cast<float *>(input_blob->GetHandle().base);
    float *output_ptr        = static_cast<float *>(output_blob->GetHandle().base);
    float *ccm_weight        = resource->ccm_weight_handle.force_to<float *>();
    float *ccm_bias          = resource->ccm_bias_handle.force_to<float *>();
    float *shifts            = resource->shifts_handle.force_to<float *>();
    float *slopes            = resource->slopes_handle.force_to<float *>();
    float *projection_weight = resource->projection_weight_handle.force_to<float *>();
    float *projection_bias   = resource->projection_bias_handle.force_to<float *>();

    DimsVector output_dims   = output_blob->GetBlobDesc().dims;
    DimsVector input_dims    = input_blob->GetBlobDesc().dims;
    int batch                = input_dims[0];
    int channel              = input_dims[1];
    int height               = input_dims[2];
    int width                = input_dims[3];
    int input_channel_stride = height * width;
    int input_batch_stride   = channel * input_channel_stride;
    int output_batch_stride  = input_channel_stride * 1;

    for (int b = 0; b < batch; ++b) {
        const float *ptr0 = input_ptr + input_batch_stride * b;
        const float *ptr1 = input_ptr + input_batch_stride * b + input_channel_stride;
        const float *ptr2 = input_ptr + input_batch_stride * b + 2 * input_channel_stride;
        float *guide_out  = output_ptr + output_batch_stride * b;

        for (int y = 0; y < height; y++) {
            int x = 0;
            for (; x < width; x++) {
                float r = ptr0[0];
                float g = ptr1[0];
                float b = ptr2[0];

                // use ccm, create new r, g, b value
                float new_r = ccm_weight[0] * r + ccm_weight[1] * g + ccm_weight[2] * b + ccm_bias[0];
                float new_g = ccm_weight[3] * r + ccm_weight[4] * g + ccm_weight[5] * b + ccm_bias[1];
                float new_b = ccm_weight[6] * r + ccm_weight[7] * g + ccm_weight[8] * b + ccm_bias[2];

                // use slope and shifts per channel
                float guide_r = 0;
                float guide_g = 0;
                float guide_b = 0;
                for (int i = 0; i < 4; i++) {
                    guide_r += slopes[i + 0] * std::max(new_r - shifts[i + 0], float(0));
                    guide_g += slopes[i + 4] * std::max(new_g - shifts[i + 4], float(0));
                    guide_b += slopes[i + 8] * std::max(new_b - shifts[i + 8], float(0));
                }

                // channel mix
                float guide_value = projection_weight[0] * guide_r + projection_weight[1] * guide_g +
                                    projection_weight[2] * guide_b + projection_bias[0];

                guide_out[0] = std::min(std::max(guide_value, 0.f), 1.f);

                ptr0 += 1;
                ptr1 += 1;
                ptr2 += 1;
                guide_out += 1;
            }
        }
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(HdrGuide, LAYER_HDRGUIDE);

}  // namespace TNN_NS
