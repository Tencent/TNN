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
#if TNN_ARM82
#include "tnn/device/arm/acc/deconvolution/arm_deconv_fp16_layer_common.h"
#include "tnn/device/arm/acc/compute/gemm_function.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"

#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
/*
ArmDeconvFp16LayerCommon as the last solution, always return true
handle the case group != 1, dilate != 1, any pads and strides
*/
bool ArmDeconvFp16LayerCommon::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    return true;
}

ArmDeconvFp16LayerCommon::~ArmDeconvFp16LayerCommon() {}

Status ArmDeconvFp16LayerCommon::allocateBufferWeight(const std::vector<Blob *> &inputs,
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
        const int goc_8 = UP_DIV(goc, 8);
        const int gic_8 = UP_DIV(gic, 8);

        size_t weight_count   = group * goc_8 * gic_8 * kh * kw * 64;
        size_t data_byte_size = DataTypeUtils::GetBytesSize(DATA_TYPE_HALF);
        RawBuffer temp_buffer(weight_count * data_byte_size);
        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
            size_t weight_nchw_count = group * goc * gic * kh * kw;
            RawBuffer filter_half(weight_nchw_count * data_byte_size);
            Float2Half(filter_half.force_to<fp16_t *>(), conv_res->filter_handle.force_to<float *>(),
                       weight_nchw_count);
            // using int16_t to copy weights
            ConvertWeightsFromGIOHWToGOHWI64(filter_half.force_to<int16_t *>(), temp_buffer.force_to<int16_t *>(), group,
                                             ic, oc, conv_param->kernels[1], conv_param->kernels[0]);
        } else if (conv_res->filter_handle.GetDataType() == DATA_TYPE_HALF) {
            // soft fp16 -> fp32 -> hard fp16 TBD
            ConvertWeightsFromGIOHWToGOHWI64(conv_res->filter_handle.force_to<int16_t *>(),
                                             temp_buffer.force_to<int16_t *>(), group, ic, oc, conv_param->kernels[1],
                                             conv_param->kernels[0]);
        } else {
            LOGE("WEIGHT DATATYPE NOT SUPPORTED NOW\n");
            return Status(TNNERR_PARAM_ERR, "FP16 DECONV COMMON ONLY SUPPORT WEIGHT DATATYPE FLOAT AND HALF");
        }
        buffer_weight_ = temp_buffer;
    }
    return TNN_OK;
}

Status ArmDeconvFp16LayerCommon::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    DataType data_type       = output->GetBlobDesc().data_type;
    const size_t data_byte_size = DataTypeUtils::GetBytesSize(DATA_TYPE_HALF);

    const int batch  = dims_output[0];
    const int group  = conv_param->group;
    auto input_width = dims_input[3], input_height = dims_input[2], ic = dims_input[1],
         input_slice  = UP_DIV(dims_input[1], 8);
    auto output_width = dims_output[3], output_height = dims_output[2], oc = dims_output[1],
         output_slice = UP_DIV(dims_output[1], 8);

    auto gic                      = dims_input[1] / group;
    auto goc                      = dims_output[1] / group;
    auto gic_8                    = UP_DIV(gic, 8);
    auto goc_8                    = UP_DIV(goc, 8);
    size_t input_size_per_group  = input_width * input_height * gic_8 * 8;
    size_t output_size_per_group = output_width * output_height * goc_8 * 8;

    int kernel_x = conv_param->kernels[0];
    int kernel_y = conv_param->kernels[1];

    const fp16_t *src_origin = reinterpret_cast<const fp16_t *>(GetBlobHandlePtr(input->GetHandle()));
    fp16_t *dst_origin       = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));

    int dst_w_pad = output_width + conv_param->pads[0] + conv_param->pads[2];
    int dst_h_pad = output_height + conv_param->pads[1] + conv_param->pads[3] + 1;

    size_t i_buf_size     = group * input_size_per_group;
    size_t o_buf_size     = group * output_size_per_group;
    size_t trans_buf_size = group * (MAX(input_size_per_group, output_size_per_group));
    size_t pad_img_size   = dst_w_pad * dst_h_pad * goc_8 * 8;

    fp16_t *work_space = reinterpret_cast<fp16_t *>(
        context_->GetSharedWorkSpace((i_buf_size + o_buf_size + trans_buf_size + pad_img_size) * data_byte_size));

    const fp16_t *input_fp16 = src_origin;
    fp16_t *output_fp16      = dst_origin;

    auto i_buffer = work_space;
    auto o_buffer = i_buffer + i_buf_size;
    auto t_buffer = o_buffer + o_buf_size;
    auto p_buffer = t_buffer + trans_buf_size;

    int weight_z_step;
    int src_z_step;
    int ic_step;
    int ic_counter;
    int w_step;
#ifdef TNN_ARM82_A64
    auto DeconvFunc = DeconvFp16O8;
    int CONVOLUTION_TILED_NUMBER = 14;
    if (gic < 8) {
        weight_z_step  = kernel_y * kernel_x * gic * 8;
        src_z_step     = k_param_->iw * k_param_->ih * 1;
        ic_step = gic;
        ic_counter = gic;
        w_step = 1;
        DeconvFunc = DeconvFp16O8C1;
        CONVOLUTION_TILED_NUMBER = 16;
    } else {
        weight_z_step  = kernel_y * kernel_x * gic_8 * 64;
        src_z_step     = k_param_->iw * k_param_->ih * 8;
        ic_step = gic_8 * 8;
        ic_counter = gic_8;
        w_step = 8;
    }
#else
    auto DeconvFunc = DeconvFp16O8;
    int CONVOLUTION_TILED_NUMBER = 8;
    if (gic < 8) {
        weight_z_step  = kernel_y * kernel_x * gic * 8;
        src_z_step     = k_param_->iw * k_param_->ih * 1;
        ic_step = gic;
        ic_counter = gic;
        w_step = 1;
        DeconvFunc = DeconvFp16O8C1;
    } else {
        weight_z_step  = kernel_y * kernel_x * gic_8 * 64;
        src_z_step     = k_param_->iw * k_param_->ih * 8;
        ic_step = gic_8 * 8;
        ic_counter = gic_8;
        w_step = 8;
    }
#endif
    int dst_z_step     = k_param_->ow * k_param_->oh * 8;
    int dst_z_step_pad = dst_w_pad * dst_h_pad * 8;

    int loop   = input_width / CONVOLUTION_TILED_NUMBER;
    int remain = input_width % CONVOLUTION_TILED_NUMBER;

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        const fp16_t *input_ptr;
        fp16_t *output_ptr;

        /*
        first unpack input tensor to nchw data format
        if gic < 8, use nchw as input
        if gic > 8, pack data to make sure every group channel algin8
        */
        if (gic < 8) {
            UnpackC8(t_buffer, input_fp16 + batch_idx * input_width * input_height * k_param_->ic_r8,
                        input_width * input_height, ic);
            input_ptr = t_buffer;
        } else if (gic_8 != (gic / 8) && group != 1) {
            UnpackC8(t_buffer, input_fp16 + batch_idx * input_width * input_height * k_param_->ic_r8,
                     input_width * input_height, ic);
            for (int g = 0; g < group; g++) {
                PackC8(i_buffer + g * input_size_per_group, t_buffer + g * input_width * input_height * gic,
                       input_width * input_height, gic);
            }
            input_ptr = i_buffer;
        } else {
            input_ptr = input_fp16 + batch_idx * input_width * input_height * k_param_->ic_r8;
        }

        if (goc_8 != (goc / 8) && group != 1) {
            output_ptr = o_buffer;
        } else {
            output_ptr = output_fp16 + batch_idx * output_width * output_height * k_param_->oc_r8;
        }

        for (int g = 0; g < group; g++) {
            auto input_g_ptr  = input_ptr + g * input_width * input_height * ic_step;
            auto output_g_ptr = output_ptr + g * output_width * output_height * goc_8 * 8;
            auto weight_ptr   = buffer_weight_.force_to<fp16_t *>() + g * goc_8 * weight_z_step;

            // prepare init value
            memset(p_buffer, 0, pad_img_size * data_byte_size);

            OMP_PARALLEL_FOR_
            for (int z = 0; z < goc_8; z++) {
                auto weight_z = weight_ptr + z * weight_z_step;
                auto dst_z    = p_buffer + z * dst_z_step_pad;
                for (int dy = 0; dy < k_param_->ih; dy++) {
                    auto src_y = input_g_ptr + dy * k_param_->iw * w_step;
                    auto dst_y = dst_z + dy * conv_param->strides[1] * dst_w_pad * 8;
                    for (int dx = 0; dx <= loop; dx++) {
                        auto x_idx   = dx * CONVOLUTION_TILED_NUMBER;
                        auto x_count = MIN(CONVOLUTION_TILED_NUMBER, k_param_->iw - x_idx);
                        auto src_x   = input_g_ptr + dy * k_param_->iw * w_step + x_idx * w_step;
                        auto dst_x   = dst_y + x_idx * conv_param->strides[0] * 8;
                        // avoid using too much variables inside omp region when compile armv7
                        // int dilate_y_step = dst_w_pad * 8 * conv_param->dialations[1];
                        // int dilate_x_step = 8 * conv_param->dialations[0];
                        // int dst_w_step    = conv_param->strides[0] * 8;
                        DeconvFunc((fp16_t*)dst_x, (const fp16_t*)src_x, (const fp16_t*)weight_z, x_count,
                                    conv_param->strides[0] * 8, ic_counter, src_z_step, conv_param->kernels[0], conv_param->kernels[1],
                                    8 * conv_param->dialations[0], dst_w_pad * 8 * conv_param->dialations[1]);
                    }
                }
            }

            // crop inner image
            OMP_PARALLEL_FOR_
            for (int z = 0; z < goc_8; z++) {
                auto src_z = p_buffer + z * dst_z_step_pad;
                auto dst_z = output_g_ptr + z * dst_z_step;
                for (int dy = 0; dy < output_height; dy++) {
                    auto src_y = src_z + (dy + conv_param->pads[2]) * dst_w_pad * 8 + conv_param->pads[0] * 8;
                    auto dst_y = dst_z + dy * output_width * 8;
                    memcpy(dst_y, src_y, output_width * 8 * data_byte_size);
                }
            }
        }

        /*
        first unpack every group output data to get nchw data format
        pack data to make sure output tensor channel algin8 and continuously
        */
        if (goc_8 != (goc / 8) && group != 1) {
            for (int g = 0; g < group; g++) {
                UnpackC8(t_buffer + g * output_width * output_height * goc,
                         output_ptr + g * output_width * output_height * goc_8 * 8, output_width * output_height, goc);
            }
            PackC8(output_fp16 + batch_idx * output_width * output_height * k_param_->oc_r8, t_buffer,
                   output_width * output_height, oc);
        }
    }

    PostExec<fp16_t>(outputs);

    return TNN_OK;
}

}  // namespace TNN_NS
#endif
