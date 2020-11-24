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

#include "tnn/device/arm/acc/arm_upsample_layer_acc.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

static inline int upsample_nearest2d(float *output_data, const float *input_data, int ih, int iw, int oh, int ow,
                                     int c_4) {
    auto src_z_step = iw * ih * 4;
    auto dst_z_step = ow * oh * 4;

    const float height_scale = (float)ih / (float)oh;
    const float width_scale  = (float)iw / (float)ow;

    OMP_PARALLEL_FOR_
    for (int z = 0; z < c_4; z++) {
        auto dst_z = output_data + z * dst_z_step;
        auto src_z = input_data + z * src_z_step;
        for (int h = 0; h < oh; h++) {
            int scale_h = h * height_scale;
            auto dst_y  = dst_z + h * ow * 4;
            auto src_y  = src_z + scale_h * iw * 4;
            for (int w = 0; w < ow; w++) {
                int scale_w = w * width_scale;
                Float4::save(dst_y + w * 4, Float4::load(src_y + scale_w * 4));
            }
        }
    }

    return 0;
}

static inline int upsample_bilinear2d(float *output_data, const float *input_data, int ih, int iw, int oh, int ow,
                                      int c_4, bool align_corners) {
    auto src_z_step = iw * ih * 4;
    auto dst_z_step = ow * oh * 4;
    auto src_y_step = iw * 4;

    RawBuffer h_coeffs(oh * sizeof(float));
    RawBuffer w_coeffs(ow * sizeof(float));
    auto h_coeffs_ptr = h_coeffs.force_to<float *>();
    auto w_coeffs_ptr = w_coeffs.force_to<float *>();

    if (align_corners) {
        const float rheight = (oh > 1) ? (float)(ih - 1) / (oh - 1) : 0.f;
        const float rwidth  = (ow > 1) ? (float)(iw - 1) / (ow - 1) : 0.f;
        for (int h = 0; h < oh; ++h) {
            h_coeffs_ptr[h] = h * rheight;
        }
        for (int w = 0; w < ow; ++w) {
            w_coeffs_ptr[w] = w * rwidth;
        }
    } else {
        const float rheight = (oh > 1) ? (float)(ih) / (oh) : 0.f;
        const float rwidth  = (ow > 1) ? (float)(iw) / (ow) : 0.f;
        for (int h = 0; h < oh; ++h) {
            h_coeffs_ptr[h] = rheight * (h + 0.5) - 0.5;
            h_coeffs_ptr[h] = h_coeffs_ptr[h] >= 0 ? h_coeffs_ptr[h] : 0;
        }
        for (int w = 0; w < ow; ++w) {
            w_coeffs_ptr[w] = rwidth * (w + 0.5) - 0.5;
            w_coeffs_ptr[w] = w_coeffs_ptr[w] >= 0 ? w_coeffs_ptr[w] : 0;
        }
    }

    OMP_PARALLEL_FOR_
    for (int h2 = 0; h2 < oh; ++h2) {
        const float h1r      = h_coeffs_ptr[h2];
        const int h1         = h1r;
        const int h1p        = (h1 < ih - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = (float)1. - h1lambda;
        for (int w2 = 0; w2 < ow; ++w2) {
            const float w1r      = w_coeffs_ptr[w2];
            const int w1         = w1r;
            const int w1p        = (w1 < iw - 1) ? 1 : 0;
            const float w1lambda = w1r - w1;
            const float w0lambda = (float)1. - w1lambda;
            const float *Xdata   = &(input_data[h1 * iw * 4 + w1 * 4]);
            float *Ydata         = &(output_data[h2 * ow * 4 + w2 * 4]);
            for (int z = 0; z < c_4; z++) {
                Float4::save(Ydata,
                             (Float4::load(Xdata) * w0lambda + Float4::load(Xdata + w1p * 4) * w1lambda) * h0lambda +
                                 (Float4::load(Xdata + h1p * src_y_step) * w0lambda +
                                  Float4::load(Xdata + h1p * src_y_step + w1p * 4) * w1lambda) *
                                     h1lambda);

                Xdata += src_z_step;
                Ydata += dst_z_step;
            }
        }
    }

    return 0;
}

ArmUpsampleLayerAcc::~ArmUpsampleLayerAcc() {}

Status ArmUpsampleLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);

    DataType data_type = outputs[0]->GetBlobDesc().data_type;

    if (data_type == DATA_TYPE_INT8 && !buffer_scale_.GetBytesSize()) {
        auto dims_output    = outputs[0]->GetBlobDesc().dims;
        int total_byte_size = ROUND_UP(dims_output[1], 4) * sizeof(float);

        auto input_resource  = reinterpret_cast<BlobInt8 *>(inputs[0])->GetIntResource();
        auto output_resource = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource();
        const float *i_scale = input_resource->scale_handle.force_to<float *>();
        const float *o_scale = output_resource->scale_handle.force_to<float *>();
        int scale_len_i      = input_resource->scale_handle.GetDataCount();
        int scale_len_o      = output_resource->scale_handle.GetDataCount();

        RawBuffer temp_buffer(total_byte_size);
        float *temp_ptr = temp_buffer.force_to<float *>();
        for (int i = 0; i < dims_output[1]; i++) {
            int scale_idx_i = scale_len_i == 1 ? 0 : i;
            int scale_idx_o = scale_len_o == 1 ? 0 : i;
            if (o_scale[scale_idx_o] >= FLT_MIN)
                temp_ptr[i] = i_scale[scale_idx_i] / o_scale[scale_idx_o];
            else
                temp_ptr[i] = 0.0;
        }
        buffer_scale_ = temp_buffer;
    }

    if (data_type == DATA_TYPE_INT8 && !buffer_ones_.GetBytesSize()) {
        auto dims_output    = outputs[0]->GetBlobDesc().dims;
        int total_byte_size = ROUND_UP(dims_output[1], 4) * sizeof(float);

        RawBuffer temp_buffer(total_byte_size);
        float *temp_ptr = temp_buffer.force_to<float *>();
        for (int i = 0; i < dims_output[1]; i++) {
            temp_ptr[i] = 1.0;
        }
        buffer_ones_ = temp_buffer;
    }

    return TNN_OK;
}

Status ArmUpsampleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        auto dims_input = inputs[0]->GetBlobDesc().dims;
        int workspace_byte_size =
            dims_input[0] * ROUND_UP(dims_input[1], 4) * dims_input[2] * dims_input[3] * sizeof(float);
        if (buffer_input_fp32_.GetBytesSize() < workspace_byte_size) {
            buffer_input_fp32_ = RawBuffer(workspace_byte_size);
        }
        auto dims_output = outputs[0]->GetBlobDesc().dims;
        workspace_byte_size =
            dims_output[0] * ROUND_UP(dims_output[1], 4) * dims_output[2] * dims_output[3] * sizeof(float);
        if (buffer_output_fp32_.GetBytesSize() < workspace_byte_size) {
            buffer_output_fp32_ = RawBuffer(workspace_byte_size);
        }
    }
    return TNN_OK;
}

Status ArmUpsampleLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<UpsampleLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    DataType data_type = outputs[0]->GetBlobDesc().data_type;

    float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

    auto oc_4 = UP_DIV(dims_output[1], 4);

    if (data_type == DATA_TYPE_INT8) {
        const float *scale = buffer_scale_.force_to<float *>();
        auto workspace     = buffer_input_fp32_.force_to<float *>();
        Int8ToFloat(workspace, reinterpret_cast<int8_t *>(input_data), scale, dims_input[0], dims_input[1],
                    dims_input[2] * dims_input[3]);
        input_data  = workspace;
        output_data = buffer_output_fp32_.force_to<float *>();
    }

    if (dims_input[2] == dims_output[2] && dims_input[3] == dims_output[3]) {
        if (output_data != input_data) {
            memcpy(output_data, input_data, oc_4 * dims_input[2] * dims_input[3] * 4 * sizeof(float));
        }
    } else if (param->mode == 1) {  // nearest
        upsample_nearest2d(output_data, input_data, dims_input[2], dims_input[3], dims_output[2], dims_output[3], oc_4);
    } else if (param->mode == 2) {  // bilinear/linear
        upsample_bilinear2d(output_data, input_data, dims_input[2], dims_input[3], dims_output[2], dims_output[3], oc_4,
                            (bool)param->align_corners);
    } else {
        LOGE("Error: Upsample dont support resize mode\n");
        return Status(TNNERR_MODEL_ERR, "Error: Upsample dont support resize mode");
    }

    if (data_type == DATA_TYPE_INT8) {
        const float *scale = buffer_ones_.force_to<float *>();
        auto workspace     = output_data;
        output_data        = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
        FloatToInt8(reinterpret_cast<int8_t *>(output_data), workspace, scale, dims_output[0], dims_output[1],
                    dims_output[2] * dims_output[3]);
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Upsample, LAYER_UPSAMPLE)

}  // namespace TNN_NS
