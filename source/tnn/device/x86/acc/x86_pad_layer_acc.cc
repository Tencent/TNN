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

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/x86_device.h"

namespace TNN_NS {

DECLARE_X86_ACC(Pad, LAYER_PAD);

#define GetPadCommonParams                                          \
    int pad_l = layer_param->pads[0];                               \
    int pad_r = layer_param->pads[1];                               \
    int pad_t = layer_param->pads[2];                               \
    int pad_b = layer_param->pads[3];                               \
    int pad_c_b = layer_param->pads[4];                               \
    int pad_c_e = layer_param->pads[5];                               \

Status X86_CONST_PAD(float *input_data, float *output_data, const int batch,
                     const int input_channel, const int input_height, const int input_width,
                     const int output_channel,const int output_height,const int output_width,
                     PadLayerParam* layer_param) {
    GetPadCommonParams;
    const float value = layer_param->value;

    int cb_border = pad_c_b;
    int ce_border = pad_c_b + input_channel;
    int hb_border = pad_t;
    int he_border = pad_t + input_height;
    int wb_border = pad_l;
    int we_border = pad_l + input_width;

    for (int n = 0; n < batch; n++) {
        auto output_data_ptr = output_data + n * output_channel * output_height * output_width;
        for (int idx = 0; idx < cb_border * output_height * output_width; idx++) { // c_b
            output_data_ptr[idx] = value;
        }
        output_data_ptr += cb_border * output_height * output_width;
        for (int c = cb_border; c < ce_border; c++) {
            auto input_data_ptr = input_data + (n * input_channel + c - cb_border) * input_height * input_width;
            for (int h = 0; h < hb_border * output_width; h++) { // h_b
                output_data_ptr[h] = value;
            }
            for (int h = hb_border; h < he_border; h++) { // h_center
                for (int w = 0; w < wb_border; w++) { // w_b
                    output_data_ptr[h * output_width + w] = value;
                }
                memcpy(output_data_ptr + h * output_width + wb_border, 
                        input_data_ptr + (h - hb_border) * input_width,
                        input_width * sizeof(float));
                for (int w = we_border; w < output_width; w++) { // w_e
                    output_data_ptr[h * output_width + w] = value;
                }
            }
            for (int h = he_border * output_width; h < output_height * output_width; h++) { // h_e
                output_data_ptr[h] = value;
            }
            output_data_ptr += output_height * output_width;
        }
        for (int idx = 0; idx < pad_c_e * output_height * output_width; idx++) { // c_e
            output_data_ptr[idx] = value;
        }
    }

    return TNN_OK;
}

Status X86_REFELCT_PAD(float *input_data, float *output_data, const int batch,
                       const int input_channel, const int input_height, const int input_width,
                       const int output_channel,const int output_height,const int output_width,
                       PadLayerParam* layer_param) {
    GetPadCommonParams;

    for (int c = 0; c < batch * output_channel; c++) {
        auto input_data_ptr = input_data + c * input_height * input_width;
        auto output_data_ptr = output_data + c * output_height * output_width;

        // center
        for (int h = 0; h < input_height; h++) {
            auto output_ptr_h = output_data_ptr + (h + pad_t) * output_width;
            auto input_ptr_h  = input_data_ptr + h * input_width;
            for (int i = 0; i < pad_l; i++) {
                output_ptr_h[i] = input_ptr_h[pad_l - i];
            }
            memcpy(output_ptr_h + pad_l, input_ptr_h, input_width * sizeof(float));
            for (int i = 0; i < pad_r; i++) {
                output_ptr_h[i + pad_l + input_width] = input_ptr_h[input_width - i - 2];
            }
        }

        // top
        for (int h = 0; h < pad_t; h++) {
            auto output_ptr_h = output_data_ptr + output_width * h;
            auto output_ref_h = output_data_ptr + output_width * (pad_t + pad_t - h);
            memcpy(output_ptr_h, output_ref_h, output_width * sizeof(float));
        }

        // bottom
        for (int h = 0; h < pad_b; h++) {
            auto output_ptr_h = output_data_ptr + output_width * (h + input_height + pad_t);
            auto output_ref_h = output_data_ptr + output_width * (input_height + pad_t - 2 - h);
            memcpy(output_ptr_h, output_ref_h, output_width * sizeof(float));
        }  
    }

    return TNN_OK;
}

Status X86PadLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    auto param = dynamic_cast<PadLayerParam *>(param_);

    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];

    auto input_dim   = input_blob->GetBlobDesc().dims;
    auto output_dim  = output_blob->GetBlobDesc().dims;

    int batch               = output_dim[0];
    int channels            = output_dim[0] * output_dim[1];
    int output_channel      = output_dim[1];
    int output_height       = output_dim[2];
    int output_width        = output_dim[3];
    int input_channel       = input_dim[1];
    int input_height        = input_dim[2];
    int input_width         = input_dim[3];

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = handle_ptr<float *>(input_blob->GetHandle());
        float *output_data = handle_ptr<float *>(output_blob->GetHandle());

        if (param->type == 0) {
            // mode: const
            X86_CONST_PAD(input_data, output_data, batch,
                          input_channel, input_height, input_width,
                          output_channel, output_height, output_width, param);
        } else if (param->type == 1) {
            // mode: reflect
            X86_REFELCT_PAD(input_data, output_data, batch,
                            input_channel, input_height, input_width,
                            output_channel, output_height, output_width, param);
        } else {
            LOGE("Error: layer param is not supported: type:%d\n", param->type);
            return Status(TNNERR_PARAM_ERR, "Error: layer param is not supported");
        }
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_X86_ACC(Pad, LAYER_PAD);

}   // namespace TNN_NS