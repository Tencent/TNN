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

#include "tnn/device/arm/acc/arm_signed_mul_layer_acc.h"

namespace TNN_NS {

Status ArmSignedMulLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto in_data_type = inputs[0]->GetBlobDesc().data_type;
    if (in_data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (in_data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    } else {
        return TNNERR_LAYER_ERR;
    }
}

template <typename T>
Status ArmSignedMulLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<SignedMulLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: SignedMulLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: SignedMulLayerParam is nil");
    }

    auto alpha     = layer_param->alpha;
    auto beta      = layer_param->beta;
    auto gamma_inv = 1.0f / layer_param->gamma;

    Float4 alpha_4 = Float4(alpha);
    Float4 val_gt4 = Float4((beta + 1.0) * gamma_inv / 2.0);
    Float4 val_lt4 = Float4((beta - 1.0) * gamma_inv / 2.0);

    auto input_blob    = inputs[0];
    auto output_blob   = outputs[0];
    T *input_data  = reinterpret_cast<T *>(GetBlobHandlePtr(input_blob->GetHandle()));
    T *output_data = reinterpret_cast<T *>(GetBlobHandlePtr(output_blob->GetHandle()));
    int batch          = input_blob->GetBlobDesc().dims[0];
    int channel        = input_blob->GetBlobDesc().dims[1];
    int channel_r4     = UP_DIV(channel, 4);
    int channel_size   = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);

    for (int b = 0; b < batch; b++) {
        T *input_data_c  = input_data  + b * channel_r4 * channel_size * 4;
        T *output_data_c = output_data + b * channel_r4 * channel_size * 4;
        for (int c = 0; c < channel_r4; c++) {
            for (int i = 0; i < channel_size; i++) {
                Float4 val  = Float4::load(input_data_c);
                Float4 res1 = Float4::bsl_cgt(val, alpha_4, val_gt4, val_lt4);
                Float4 res2 = Float4::bsl_clt(val, alpha_4, val_lt4, val_gt4);
                Float4 res  = res1 + res2;
                Float4::save(output_data_c, res);
                input_data_c  += 4;
                output_data_c += 4;
            }
        }

        for (int c = channel_r4 - 1; c >= 0; c--) {
            T *output_data_c  = output_data + (b * channel_r4 + c) * channel_size * 4;
            T *output_data_c0 = output_data + b * channel_r4 * channel_size * 4;
            for (int i = 0; i < channel_size; i++) {
                Float4 val = Float4::load(output_data_c);
                Float4 res = val * output_data_c0[0];
                Float4::save(output_data_c, res);
                output_data_c  += 4;
                output_data_c0 += 4;
            }
        }
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(SignedMul, LAYER_SIGNED_MUL);
REGISTER_ARM_LAYOUT(LAYER_SIGNED_MUL, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
