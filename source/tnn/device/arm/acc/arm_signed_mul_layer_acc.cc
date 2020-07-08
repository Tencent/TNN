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

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include <iostream>

namespace TNN_NS {

DECLARE_ARM_ACC(SignedMul, LAYER_SIGNED_MUL);

Status ArmSignedMulLayerAcc::DoForward(const std::vector<tnn::Blob *> &inputs, const std::vector<tnn::Blob *> &outputs) {
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
    int channel_r4     = UP_DIV(channel, 4);

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channel_r4; c++) {
            int channel_index = b * channel_r4 + c;
            float *input_data_c_r4 = input_data + channel_index * channel_size * 4;
            float *output_data_c_r4 = output_data + channel_index * channel_size * 4;
            for (int i = 0; i < channel_size; i++) {
                float *input_data_c = input_data_c_r4 + i * 4;
                float *output_data_c = output_data_c_r4 + i * 4;
                for (int j = 0; j < 4; j++) {
                    //sub 
                    float temp = input_data_c[j] - alpha;

                    //sign
                    if (temp > 0) {
                        temp = 1;
                    } else if (temp < 0) {
                        temp = -1;
                    }

                    //add
                    temp += beta;

                    //div
                    temp *= gamma_inv;
                    output_data_c[j] = temp;
                }
            }
        }

        //mul
        float *output_data_c0_r4 = output_data + b * channel_r4 * channel_size * 4;
        for (int c = channel_r4 - 1; c >= 0; c--) {
            int channel_index = b * channel_r4 + c;
            float *output_data_c_r4 = output_data + channel_index * channel_size * 4;
            for (int i = 0; i < channel_size; i++) {
                float *output_data_c = output_data_c_r4 + i * 4;
                float *output_data_c0 = output_data_c0_r4 + i * 4;
                for (int j = 3; j >= 0; j--) {
                    output_data_c[j] *= output_data_c0[0];
                }
            }
        }
    }

    return 0;
}

REGISTER_ARM_ACC(SignedMul, LAYER_SIGNED_MUL);

}  // namespace TNN_NS
