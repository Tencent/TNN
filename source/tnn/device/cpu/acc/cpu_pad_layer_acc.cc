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

namespace TNN_NS {

DECLARE_CPU_ACC(Pad, LAYER_PAD);

Status CpuPadLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

#define GetPadCommonParams                                              \
    int pad_l = layer_param->pads[0];                                   \
    int pad_r = layer_param->pads[1];                                   \
    int pad_t = layer_param->pads[2];                                   \
    int pad_b = layer_param->pads[3];                                   \
    int pad_c_b = layer_param->pads[4];                                 \
    int pad_c_e = layer_param->pads[5];

void ConstPad(float* input_data, float* output_data, const int batch,
              const int input_channel, const int input_height, const int input_width,
              const int output_channel, const int output_height, const int output_width,
              PadLayerParam* layer_param) {
    GetPadCommonParams;
    float value = layer_param->value;

    int cb_border = pad_c_b;
    int ce_border = pad_c_b + input_channel;
    int ht_border = pad_t;
    int hb_border = pad_t + input_height;
    int wl_border = pad_l;
    int wr_border = pad_l + input_width;
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < output_channel; c++) {
            auto input_data_ptr  = input_data + (n * input_channel + c - cb_border) * input_height * input_width;
            auto output_data_ptr = output_data + (n * output_channel + c) * output_height * output_width;
            for (int h = 0; h < output_height; ++h) {
                for (int w = 0; w < output_width; ++w) {
                    if (c < cb_border || c >= ce_border ||
                        h < ht_border || h >= hb_border ||
                        w < wl_border || w >= wr_border ) {
                        output_data_ptr[h * output_width + w] = value;
                    } else {
                        int output_idx              = h * output_width + w;
                        int input_idx               = (h - ht_border) * input_width + w - wl_border;
                        output_data_ptr[output_idx] = input_data_ptr[input_idx];
                    }
                }
            }
        }
    }
}

void ReflectPad(float* input_data, float* output_data, const int batch,
                const int input_channel, const int input_height, const int input_width,
                const int output_channel, const int output_height, const int output_width,
                const int channels, const int input_width_bytes, const int data_byte_size,
                PadLayerParam* layer_param) {
    GetPadCommonParams;

    for (int c = 0; c < channels; c++) {
        auto input_data_ptr  = input_data + c * input_height * input_width;
        auto output_data_ptr = output_data + c * output_height * output_width;

        // center
        for (int h = 0; h < input_height; h++) {
            auto output_ptr_h = output_data_ptr + output_width * (h + pad_t);
            auto input_ptr_h  = input_data_ptr + input_width * h;
            for (int i = 0; i < pad_l; i++) {
                output_ptr_h[i] = input_ptr_h[pad_l - i];
            }
            memcpy(output_ptr_h + pad_l, input_ptr_h, input_width_bytes);
            for (int i = 0; i < pad_r; i++) {
                output_ptr_h[i + pad_l + input_width] = input_ptr_h[input_width - i - 2];
            }
        }

        // top
        for (int h = 0; h < pad_t; h++) {
            auto output_ptr_h = output_data_ptr + output_width * h;
            auto output_ref_h = output_data_ptr + output_width * (pad_t + pad_t - h);
            memcpy(output_ptr_h, output_ref_h, output_width * data_byte_size);
        }

        // bottom
        for (int h = 0; h < pad_b; h++) {
            auto output_ptr_h = output_data_ptr + output_width * (h + input_height + pad_t);
            auto output_ref_h = output_data_ptr + output_width * (input_height + pad_t - 2 - h);
            memcpy(output_ptr_h, output_ref_h, output_width * data_byte_size);
        }
    }
}

Status EdgePad(float* input_data, float* output_data, const int batch,
               const int input_channel, const int input_height, const int input_width,
               const int output_channel, const int output_height, const int output_width,
               const int channels, const int input_width_bytes, const int data_byte_size,
               PadLayerParam* layer_param) {
    GetPadCommonParams;

    for (int c = 0; c < channels; c++) {
        auto input_data_ptr  = input_data + c * input_height * input_width;
        auto output_data_ptr = output_data + c * output_height * output_width;

        int ht_border = pad_t;
        int hb_border = pad_t + input_height;
        int wl_border = pad_l;
        int wr_border = pad_l + input_width;
        // top
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                if (h < ht_border && w < wl_border) {
                    // left_top
                    output_data_ptr[h * output_width + w] = input_data_ptr[0 * input_width + 0];
                } else if (h < ht_border && w >= wr_border) {
                    // right_top
                    output_data_ptr[h * output_width + w] = input_data_ptr[0 * input_width + input_width - 1];
                } else if (h > hb_border && w < wl_border) {
                    // left_bottom
                    output_data_ptr[h * output_width + w] =
                        input_data_ptr[(input_height - 1) * input_width + 0];
                } else if (h > hb_border && w >= wr_border) {
                    // right_bottom
                    output_data_ptr[h * output_width + w] =
                        input_data_ptr[(input_height - 1) * input_width + input_width - 1];
                } else if (h >= ht_border && w < wl_border) {
                    // left
                    output_data_ptr[h * output_width + w] = input_data_ptr[(h - ht_border) * input_width + 0];
                } else if (h >= ht_border && w >= wr_border) {
                    // right
                    output_data_ptr[h * output_width + w] =
                        input_data_ptr[(h - ht_border) * input_width + input_width - 1];
                } else if (h < ht_border && w < wr_border) {
                    // top
                    output_data_ptr[h * output_width + w] = input_data_ptr[0 * input_width + w];
                } else if (h >= hb_border && w < wr_border) {
                    // bottom
                    output_data_ptr[h * output_width + w] =
                        input_data_ptr[(input_height - 1) * input_width + w];
                } else {
                    LOGE("Error: Stuck in the wrong branch: type:%d\n", layer_param->type);
                    return Status(TNNERR_PARAM_ERR, "Error: layer param is not supported");
                }
            }
        }
    }

    return TNN_OK;
}

Status CpuPadLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PadLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: layer param is nil");
    }
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];

    auto input_dims  = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;

    int batch                   = output_dims[0];
    int channels                = output_dims[0] * output_dims[1];
    int output_channel          = output_dims[1];
    int output_height           = output_dims[2];
    int output_width            = output_dims[3];
    int input_channel           = input_dims[1];
    int input_height            = input_dims[2];
    int input_width             = input_dims[3];
    int data_byte_size          = DataTypeUtils::GetBytesSize(input_blob->GetBlobDesc().data_type);
    const int input_width_bytes = input_width * data_byte_size;

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT ||
        output_blob->GetBlobDesc().data_type == DATA_TYPE_INT32 ||
        output_blob->GetBlobDesc().data_type == DATA_TYPE_UINT32) {
        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);

        if (layer_param->type == 0) {
            // mode: const
            ConstPad(input_data, output_data, batch, input_channel, input_height, input_width,
                     output_channel, output_height, output_width, layer_param);
        } else if (layer_param->type == 1) {
            // mode: reflect
            ReflectPad(input_data, output_data, batch, input_channel, input_height, input_width,
                       output_channel, output_height, output_width, channels, input_width_bytes,
                       data_byte_size, layer_param);
        } else if (layer_param->type == 2) {
            // mode: edge
            Status status = EdgePad(
                    input_data, output_data, batch, input_channel, input_height, input_width,
                    output_channel, output_height, output_width, channels, input_width_bytes,
                    data_byte_size, layer_param);
            if (status != TNN_OK) {
                return status;
            }
        } else {
            LOGE("Error: layer param is not supported: type:%d\n", layer_param->type);
            return Status(TNNERR_PARAM_ERR, "Error: layer param is not supported");
        }
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Pad, LAYER_PAD);

}  // namespace TNN_NS
