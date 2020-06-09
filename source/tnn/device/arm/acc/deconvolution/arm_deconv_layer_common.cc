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

#include "tnn/device/arm/acc/deconvolution/arm_deconv_layer_common.h"
#include "tnn/device/arm/acc/compute/gemm_function.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"

#include "tnn/utils/omp_utils.h"

#if defined(__aarch64__)
#define CONVOLUTION_TILED_NUMBER (14)
#else
#define CONVOLUTION_TILED_NUMBER (8)
#endif

namespace TNN_NS {
/*
ArmDeconvLayerCommonas as the last solution, always return true
handle the case group != 1, dilate != 1, any pads and strides
*/
bool ArmDeconvLayerCommon::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    return true;
}

ArmDeconvLayerCommon::~ArmDeconvLayerCommon() {}

Status ArmDeconvLayerCommon::allocateBufferWeight(const std::vector<Blob *> &inputs,
                                                  const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    if (!buffer_weight_.GetBytesSize()) {
        int kw          = conv_param->kernels[0];
        int kh          = conv_param->kernels[1];
        const int ic    = inputs[0]->GetBlobDesc().dims[1];
        const int oc    = outputs[0]->GetBlobDesc().dims[1];
        const int group = conv_param->group;
        const int goc   = oc / group;
        const int gic   = ic / group;
        const int goc_4 = UP_DIV(goc, 4);
        const int gic_4 = UP_DIV(gic, 4);

        const float *src = conv_res->filter_handle.force_to<float *>();
        CHECK_PARAM_NULL(src);

        int weight_count   = group * goc_4 * gic_4 * kh * kw * 16;
        int data_byte_size = DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());

        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT ||
            conv_res->filter_handle.GetDataType() == DATA_TYPE_INT8) {
            RawBuffer temp_buffer(weight_count * data_byte_size);
            float *dst = temp_buffer.force_to<float *>();

            if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
                ConvertWeightsFromGIOHWToGOHWI16((float *)src, (float *)dst, group, ic, oc, conv_param->kernels[1],
                                                 conv_param->kernels[0]);
            } else {
                // Todo
            }

            buffer_weight_ = temp_buffer;
        } else {
            LOGE("Error: DataType %d not support\n", conv_res->filter_handle.GetDataType());
            return Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        }
    }
    return TNN_OK;
}

Status ArmDeconvLayerCommon::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    }
    return TNNERR_LAYER_ERR;
}

template <typename T>
Status ArmDeconvLayerCommon::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    DataType data_type       = output->GetBlobDesc().data_type;
    const int data_byte_size = 4;

    const int batch  = dims_output[0];
    const int group  = conv_param->group;
    auto input_width = dims_input[3], input_height = dims_input[2], ic = dims_input[1],
         input_slice  = UP_DIV(dims_input[1], 4);
    auto output_width = dims_output[3], output_height = dims_output[2], oc = dims_output[1],
         output_slice = UP_DIV(dims_output[1], 4);

    auto gic                    = dims_input[1] / group;
    auto goc                    = dims_output[1] / group;
    auto gic_4                  = UP_DIV(gic, 4);
    auto goc_4                  = UP_DIV(goc, 4);
    auto input_bytes_per_group  = input_width * input_height * gic_4 * 4 * data_byte_size;
    auto output_bytes_per_group = output_width * output_height * goc_4 * 4 * data_byte_size;

    int kernel_x = conv_param->kernels[0];
    int kernel_y = conv_param->kernels[1];

    T *src_origin = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    T *dst_origin = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    int dst_w_pad = output_width + conv_param->pads[0] + conv_param->pads[2];
    int dst_h_pad = output_height + conv_param->pads[1] + conv_param->pads[3] + 1;

    int i_buf_size     = group * input_bytes_per_group;
    int o_buf_size     = group * output_bytes_per_group;
    int trans_buf_size = group * (MAX(input_bytes_per_group, output_bytes_per_group));
    int pad_img_size   = dst_w_pad * dst_h_pad * goc_4 * 4 * data_byte_size;
    int bf16_buf_size  = data_type == DATA_TYPE_BFP16 ? (i_buf_size + o_buf_size) * batch : 0;

    float *work_space = reinterpret_cast<float *>(
        context_->GetSharedWorkSpace(trans_buf_size + i_buf_size + o_buf_size + pad_img_size + bf16_buf_size));
    float *input_float, *output_float;

    // bfp16 format has no separate process, convert to float
    if (data_type == DATA_TYPE_BFP16) {
        input_float = work_space;
        FloatConvert(reinterpret_cast<bfp16_t *>(src_origin), input_float,
                     batch * input_width * input_height * k_param_->ic_r4 / 4);
        output_float = input_float + i_buf_size * batch / sizeof(float);
        work_space   = output_float + o_buf_size * batch / sizeof(float);
    } else {
        input_float  = reinterpret_cast<float *>(src_origin);
        output_float = reinterpret_cast<float *>(dst_origin);
    }

    auto i_buffer = work_space;
    auto o_buffer = i_buffer + i_buf_size / data_byte_size;
    auto t_buffer = o_buffer + o_buf_size / data_byte_size;
    auto p_buffer = t_buffer + trans_buf_size / data_byte_size;

    int weight_z_step  = kernel_y * kernel_x * gic_4 * 16;
    int src_z_step     = k_param_->iw * k_param_->ih * 4;
    int dst_z_step     = k_param_->ow * k_param_->oh * 4;
    int dst_z_step_pad = dst_w_pad * dst_h_pad * 4;

    int dilate_y_step = dst_w_pad * 4 * conv_param->dialations[1];
    int dilate_x_step = 4 * conv_param->dialations[0];
    int dst_w_step    = conv_param->strides[0] * 4;

    int loop   = input_width / CONVOLUTION_TILED_NUMBER;
    int remain = input_width % CONVOLUTION_TILED_NUMBER;

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        float *input_ptr;
        float *output_ptr;

        /*
        first unpack input tensor to nchw data format
        pack data to make sure every group channel algin4
        */
        if (gic_4 != (gic / 4) && group != 1) {
            input_ptr = i_buffer;
            UnpackC4(t_buffer, input_float + batch_idx * input_width * input_height * k_param_->ic_r4,
                     input_width * input_height, ic);
            for (int g = 0; g < group; g++) {
                PackC4(input_ptr + g * input_bytes_per_group / 4, t_buffer + g * input_width * input_height * gic,
                       input_width * input_height, gic);
            }
        } else {
            input_ptr = input_float + batch_idx * input_width * input_height * k_param_->ic_r4;
        }

        if (goc_4 != (goc / 4) && group != 1) {
            output_ptr = o_buffer;
        } else {
            output_ptr = output_float + batch_idx * output_width * output_height * ROUND_UP(oc, 4);
        }

        for (int g = 0; g < group; g++) {
            auto input_g_ptr  = input_ptr + g * input_width * input_height * gic_4 * 4;
            auto output_g_ptr = output_ptr + g * output_width * output_height * goc_4 * 4;
            auto weight_ptr   = buffer_weight_.force_to<float *>() + g * goc_4 * weight_z_step;

            // prepare init value
            memset(p_buffer, 0, pad_img_size);

            OMP_PARALLEL_FOR_
            for (int z = 0; z < goc_4; z++) {
                auto weight_z = weight_ptr + z * weight_z_step;
                auto dst_z    = p_buffer + z * dst_z_step_pad;
                for (int dy = 0; dy < k_param_->ih; dy++) {
                    auto src_y = input_g_ptr + dy * k_param_->iw * 4;
                    auto dst_y = dst_z + dy * conv_param->strides[1] * dst_w_pad * 4;
                    for (int dx = 0; dx <= loop; dx++) {
                        auto x_idx   = dx * CONVOLUTION_TILED_NUMBER;
                        auto x_count = MIN(CONVOLUTION_TILED_NUMBER, k_param_->iw - x_idx);
                        auto src_x   = input_g_ptr + dy * k_param_->iw * 4 + x_idx * 4;
                        auto dst_x   = dst_y + x_idx * conv_param->strides[0] * 4;
                        DeconvFloatO4(dst_x, src_x, weight_z, x_count, dst_w_step, gic_4, src_z_step,
                                      conv_param->kernels[0], conv_param->kernels[1], dilate_x_step, dilate_y_step);
                    }
                }
            }

            // crop inner image
            OMP_PARALLEL_FOR_
            for (int z = 0; z < goc_4; z++) {
                auto src_z = p_buffer + z * dst_z_step_pad;
                auto dst_z = output_g_ptr + z * dst_z_step;
                for (int dy = 0; dy < output_height; dy++) {
                    auto src_y = src_z + (dy + conv_param->pads[2]) * dst_w_pad * 4 + conv_param->pads[0] * 4;
                    auto dst_y = dst_z + dy * output_width * 4;
                    memcpy(dst_y, src_y, output_width * 4 * data_byte_size);
                }
            }
        }

        /*
        first unpack every group output data to get nchw data format
        pack data to make sure output tensor channel algin4 and continuously
        */
        if (goc_4 != (goc / 4) && group != 1) {
            for (int g = 0; g < group; g++) {
                UnpackC4(t_buffer + g * output_width * output_height * goc,
                         output_ptr + g * output_width * output_height * goc_4 * 4, output_width * output_height, goc);
            }
            PackC4(output_float + batch_idx * output_width * output_height * ROUND_UP(oc, 4), t_buffer,
                   output_width * output_height, oc);
        }
    }

    // convert back to bfp16
    if (data_type == DATA_TYPE_BFP16) {
        FloatConvert<float, bfp16_t>(output_float, reinterpret_cast<bfp16_t *>(GetBlobHandlePtr(output->GetHandle())),
                                     batch * output_width * output_height * k_param_->oc_r4 / 4);
    }

    PostExec<T>(outputs);

    return TNN_OK;
}

}  // namespace TNN_NS
