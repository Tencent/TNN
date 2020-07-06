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

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(SignedMul, LAYER_SIGNED_MUL);

Status CpuSignedMulLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuSignedMulLayerAcc::Forward(const std::vector<tnn::Blob *> &inputs, const std::vector<tnn::Blob *> &outputs) {
    auto layer_param = dynamic_cast<SignedMulLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: SignedMulLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: SignedMulLayerParam is nil");
    }

    auto alpha     = layer_param->alpha;
    auto beta      = layer_param->beta;
    auto gamma_inv = 1.0f / layer_param->gamma;

    auto input_blob    = inputs[0];
    auto output_blob   = outputs[0];
    float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data = static_cast<float *>(output_blob->GetHandle().base);
    int batch          = input_blob->GetBlobDesc().dims[0];
    int channel        = input_blob->GetBlobDesc().dims[1];
    int channel_size   = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channel; c++) {
            int channel_index = b * channel + c;
            for (int i = 0; i < channel_size; i++) {
                int index  = channel_index * channel_size + i;
                float temp = input_data[index] - alpha;
                if (temp > 0) {
                    temp = 1;
                } else if (temp < 0) {
                    temp = -1;
                }
                temp += beta;
                temp *= gamma_inv;
                output_data[index] = temp;
            }
        }
        for (int c = channel_size - 1; c >= 0; c--) {
            int channel_index         = b * channel + c;
            int output_channel0_index = b * channel;
            for (int i = 0; i < channel_size; i++) {
                output_data[channel_index + i] *= input_data[output_channel0_index + i];
            }
        }
    }
    return 0;
}

REGISTER_CPU_ACC(SignedMul, LAYER_SIGNED_MUL);

}  // namespace TNN_NS
