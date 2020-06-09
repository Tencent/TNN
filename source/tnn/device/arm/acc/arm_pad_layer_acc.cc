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

#include "tnn/device/arm/acc/Float4.h"
#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Pad, LAYER_PAD);

Status ArmPadLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PadLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];

    auto input_dims  = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;

    int batch          = output_dims[0];
    int c_r4           = ROUND_UP(output_dims[1], 4);
    int oh             = output_dims[2];
    int ow             = output_dims[3];
    int ih             = input_dims[2];
    int iw             = input_dims[3];
    int pad_l          = layer_param->pads[0];
    int pad_r          = layer_param->pads[1];
    int pad_t          = layer_param->pads[2];
    int pad_b          = layer_param->pads[3];
    int byte_size      = DataTypeUtils::GetBytesSize(input_blob->GetBlobDesc().data_type);
    const int iw_bytes = iw * byte_size * 4;

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));
        float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));

        if (layer_param->type == 0) {
            Float4 vzero(0);
            for (int c = 0; c < batch * c_r4; c += 4) {
                auto input_ptr_c  = input_data + c * ih * iw;
                auto output_ptr_c = output_data + c * oh * ow;

                if (pad_t)
                    memset(output_ptr_c, 0, ow * pad_t * byte_size * 4);

                for (int h = 0; h < ih; ++h) {
                    auto output_ptr_h = output_ptr_c + ow * (h + pad_t) * 4;
                    auto input_ptr_h  = input_ptr_c + iw * h * 4;
                    for (int i = 0; i < pad_l; i++)
                        Float4::save(output_ptr_h + i * 4, vzero);

                    memcpy(output_ptr_h + pad_l * 4, input_ptr_h, iw_bytes);

                    for (int i = iw + pad_l; i < ow; i++)
                        Float4::save(output_ptr_h + i * 4, vzero);
                }

                if (pad_b)
                    memset(output_ptr_c + ow * (ih + pad_t) * 4, 0, ow * pad_b * byte_size * 4);
            }
        } else if (layer_param->type == 1) {
            for (int c = 0; c < batch * c_r4; c += 4) {
                auto input_ptr_c  = input_data + c * ih * iw;
                auto output_ptr_c = output_data + c * oh * ow;

                for (int h = 0; h < ih; ++h) {
                    auto output_ptr_h = output_ptr_c + ow * (h + pad_t) * 4;
                    auto input_ptr_h  = input_ptr_c + iw * h * 4;
                    for (int i = 0; i < pad_l; i++) {
                        Float4::save(output_ptr_h + i * 4, Float4::load(input_ptr_h + (pad_l - i) * 4));
                    }

                    memcpy(output_ptr_h + pad_l * 4, input_ptr_h, iw_bytes);

                    for (int i = 0; i < pad_r; i++) {
                        Float4::save(output_ptr_h + (i + pad_l + iw) * 4,
                                     Float4::load(input_ptr_h + (iw - 1 - (i + 1)) * 4));
                    }
                }

                // pad: copy from output
                for (int h = 0; h < pad_t; h++) {
                    auto output_ptr_h = output_ptr_c + ow * h * 4;
                    auto output_ref_h = output_ptr_c + ow * (pad_t + pad_t - h) * 4;
                    memcpy(output_ptr_h, output_ref_h, ow * byte_size * 4);
                }

                for (int h = 0; h < pad_b; h++) {
                    auto output_ptr_h = output_ptr_c + ow * (h + ih + pad_t) * 4;
                    auto output_ref_h = output_ptr_c + ow * (ih + pad_t - 1 - (h + 1)) * 4;
                    memcpy(output_ptr_h, output_ref_h, ow * byte_size * 4);
                }
            }
        }
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(Pad, LAYER_PAD);

}  // namespace TNN_NS
