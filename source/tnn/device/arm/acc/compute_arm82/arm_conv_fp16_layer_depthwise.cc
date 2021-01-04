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

#include "tnn/device/arm/acc/convolution/arm_conv_fp16_layer_depthwise.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"
#include "tnn/device/arm/acc/Half8.h"

namespace TNN_NS {
bool ArmConvFp16LayerDepthwise::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                       const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    if (inputs[0]->GetBlobDesc().data_type != DATA_TYPE_HALF) {
        return false;
    }

    const int group          = param->group;
    const int input_channel  = inputs[0]->GetBlobDesc().dims[1];
    const int output_channel = outputs[0]->GetBlobDesc().dims[1];

    return group == input_channel && group == output_channel && group != 1;
}

ArmConvFp16LayerDepthwise::~ArmConvFp16LayerDepthwise() {}

Status ArmConvFp16LayerDepthwise::allocateBufferWeight(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    if (!buffer_weight_.GetBytesSize()) {
        int kw = param->kernels[0];
        int kh = param->kernels[1];

        const int group  = param->group;
        const int group8 = ROUND_UP(group, 8);

        size_t weight_count = group8 * kh * kw;
        size_t data_byte_size = weight_count * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF);
        RawBuffer temp_buffer(data_byte_size + NEON_KERNEL_EXTRA_LOAD);
        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
            size_t weight_nchw_count = group * kh * kw;
            RawBuffer filter_half(weight_nchw_count * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
            Float2Half(filter_half.force_to<fp16_t *>(), conv_res->filter_handle.force_to<float *>(),
                       weight_nchw_count);
            PackC8(temp_buffer.force_to<fp16_t *>(),
                   filter_half.force_to<fp16_t *>(),
                   kh * kw, group);
        } else if (conv_res->filter_handle.GetDataType() == DATA_TYPE_HALF) {
            // soft fp16 -> fp32 -> hard fp16 TBD
            PackC8(temp_buffer.force_to<fp16_t *>(),
                   conv_res->filter_handle.force_to<fp16_t *>(),
                   kh * kw, group);
        } else {
            LOGE("Error: DataType %d not support\n", conv_res->filter_handle.GetDataType());
            return Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        }

        buffer_weight_ = temp_buffer;
    }
    return TNN_OK;
}

Status ArmConvFp16LayerDepthwise::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    const int batch    = dims_output[0];
    int dst_z_step     = k_param_->ow * k_param_->oh;
    int src_z_step     = k_param_->iw * k_param_->ih;
    int dilate_y_step  = k_param_->iw * 8 * param->dialations[1];
    int dilate_x_step  = 8 * param->dialations[0];
    int weight_z_step  = param->kernels[0] * param->kernels[1];

    int l = 0, t = 0, r = k_param_->ow, b = k_param_->oh;
    for (; l * param->strides[0] - param->pads[0] < 0; l++)
        ;
    for (; t * param->strides[1] - param->pads[2] < 0; t++)
        ;
    for (; (r - 1) * param->strides[0] - param->pads[0] + param->kernels[0] * param->dialations[0] > k_param_->iw &&
            r > l; r--)
        ;
    for (; (b - 1) * param->strides[1] - param->pads[2] + param->kernels[1] * param->dialations[1] > k_param_->ih &&
            b > t; b--)
        ;

    // lamda function to process left/right/top/bottom corner
    auto RunCorner = [=](fp16_t *dst_z, const fp16_t *src_z, const fp16_t *weight_dz, int left, int top, int right, int bottom) {
        for (int dy = top; dy < bottom; ++dy) {
            auto *dst_y        = dst_z + dy * k_param_->ow * 8;
            int srcStartY      = dy * param->strides[1] - param->pads[2];
            const auto *src_dy = src_z + srcStartY * k_param_->iw * 8;
            int sfy            = MAX(0, (UP_DIV(-srcStartY, param->dialations[1])));
            int efy            = MIN(param->kernels[1], UP_DIV(k_param_->ih - srcStartY, param->dialations[1]));
            for (int dx = left; dx < right; ++dx) {
                auto *dst_x        = dst_y + 8 * dx;
                int srcStartX      = dx * param->strides[0] - param->pads[0];
                const auto *src_dx = src_dy + srcStartX * 8;
                int sfx            = MAX(0, (UP_DIV(-srcStartX, param->dialations[0])));
                int efx            = MIN(param->kernels[0], UP_DIV(k_param_->iw - srcStartX, param->dialations[0]));
                DepthwiseUnit(dst_x,
                              src_dx + (sfx * param->dialations[0] + sfy * param->dialations[1] * k_param_->iw) * 8,
                              weight_dz + 8 * (param->kernels[0] * sfy + sfx), efx - sfx, efy - sfy,
                              8 * param->kernels[0], dilate_x_step, dilate_y_step);
            }
        }
    };

    auto *src_origin = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input->GetHandle()));
    auto *dst_origin = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto src_ptr = src_origin + batch_idx * k_param_->iw * k_param_->ih * k_param_->ic_r8;
        auto dst_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r8;

        OMP_PARALLEL_FOR_
        for (int dz = 0; dz < k_param_->oc_r8; dz += 8) {
            auto *dst_z     = dst_ptr + dst_z_step * dz;
            auto *src_z     = src_ptr + src_z_step * dz;
            auto *weight_dz = reinterpret_cast<fp16_t *>(k_param_->fil_ptr) + dz * weight_z_step;
            auto *bias_z    = reinterpret_cast<fp16_t *>(k_param_->bias) + dz;

            RunCorner(dst_z, src_z, weight_dz, 0, 0, k_param_->ow, t);
            RunCorner(dst_z, src_z, weight_dz, 0, b, k_param_->ow, k_param_->oh);
            RunCorner(dst_z, src_z, weight_dz, 0, t, l, b);
            RunCorner(dst_z, src_z, weight_dz, r, t, k_param_->ow, b);

            if (r > l && b > t) {
                DepthwiseConv(dst_z + t * k_param_->ow * 8 + l * 8,
                              src_z + (t * param->strides[1] - param->pads[2]) * k_param_->iw * 8 +
                              (l * param->strides[0] - param->pads[0]) * 8,
                              weight_dz, r - l, param->strides[0] * 8, param->kernels[0], param->kernels[1], dilate_x_step,
                              dilate_y_step, b - t, k_param_->iw * 8 * param->strides[1], k_param_->ow * 8);
            }
        }
    }
    PostExec<fp16_t>(outputs);

    return TNN_OK;
}

}  // namespace TNN_NS

#endif
