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

#include "tnn/device/arm/acc/convolution/arm_conv_layer_depthwise.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
bool ArmConvLayerDepthwise::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                       const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    const int group          = param->group;
    const int input_channel  = inputs[0]->GetBlobDesc().dims[1];
    const int output_channel = outputs[0]->GetBlobDesc().dims[1];

    return group == input_channel && group == output_channel;
}

ArmConvLayerDepthwise::~ArmConvLayerDepthwise() {}

Status ArmConvLayerDepthwise::allocateBufferWeight(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    if (!buffer_weight_.GetBytesSize()) {
        int kw = param->kernels[0];
        int kh = param->kernels[1];

        const int group  = param->group;
        const int group4 = UP_DIV(group, 4) * 4;

        const float *src = conv_res->filter_handle.force_to<float *>();

        int weight_count   = group4 * kh * kw;
        int data_byte_size = DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());

        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
            RawBuffer temp_buffer(weight_count * data_byte_size);
            float *dst = temp_buffer.force_to<float *>();

            DataFormatConverter::ConvertFromNCHWToNCHW4Float((float *)src, (float *)dst, 1, group,
                                                                param->kernels[1], param->kernels[0]);
            temp_buffer.SetDataType(DATA_TYPE_FLOAT);

            buffer_weight_ = temp_buffer;
        } else {
            LOGE("Error: DataType %d not support\n", conv_res->filter_handle.GetDataType());
            return Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        }
    }
    return TNN_OK;
}

Status ArmConvLayerDepthwise::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
Status ArmConvLayerDepthwise::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    const int batch    = dims_output[0];
    int dst_depth_quad = UP_DIV(dims_output[1], 4);
    int dst_z_step     = k_param_->ow * k_param_->oh;
    int src_z_step     = k_param_->iw * k_param_->ih;
    int dilate_y_step  = k_param_->iw * 4 * param->dialations[1];
    int dilate_x_step  = 4 * param->dialations[0];
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
    auto RunCorner = [=](T *dst_z, const T *src_z, const float *weight_dz, int left, int top, int right, int bottom) {
        for (int dy = top; dy < bottom; ++dy) {
            auto *dst_y        = dst_z + dy * k_param_->ow * 4;
            int srcStartY      = dy * param->strides[1] - param->pads[2];
            const auto *src_dy = src_z + srcStartY * k_param_->iw * 4;
            int sfy            = MAX(0, (UP_DIV(-srcStartY, param->dialations[1])));
            int efy            = MIN(param->kernels[1], UP_DIV(k_param_->ih - srcStartY, param->dialations[1]));
            for (int dx = left; dx < right; ++dx) {
                auto *dst_x        = dst_y + 4 * dx;
                int srcStartX      = dx * param->strides[0] - param->pads[0];
                const auto *src_dx = src_dy + srcStartX * 4;
                int sfx            = MAX(0, (UP_DIV(-srcStartX, param->dialations[0])));
                int efx            = MIN(param->kernels[0], UP_DIV(k_param_->iw - srcStartX, param->dialations[0]));
                DepthwiseUnit(dst_x,
                              src_dx + (sfx * param->dialations[0] + sfy * param->dialations[1] * k_param_->iw) * 4,
                              weight_dz + 4 * (param->kernels[0] * sfy + sfx), efx - sfx, efy - sfy,
                              4 * param->kernels[0], dilate_x_step, dilate_y_step);
            }
        }
    };

    auto *src_origin = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    auto *dst_origin = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    typedef void DWFunc(T *dst, const T *src, const float *weight, long width, long src_w_setup, long fw, long fh,
                        long dilateX_step, long dilateY_step, long height, long srcHStep, long dstHStep);
    DWFunc *dw_full = DepthwiseConv<T>;
    /*
    convdw3x3 stride >= 2 here
    convdw3x3s1 has separate kernel in convdws1 acc
    */
    if (param->kernels[0] == 3 && param->kernels[1] == 3) {
        dw_full = DepthwiseConv3x3<T>;
    }
    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto src_ptr = src_origin + batch_idx * k_param_->iw * k_param_->ih * k_param_->ic_r4;
        auto dst_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r4;

        OMP_PARALLEL_FOR_
        for (int dz = 0; dz < k_param_->oc_r4; dz += 4) {
            auto *dst_z     = dst_ptr + dst_z_step * dz;
            auto *src_z     = src_ptr + src_z_step * dz;
            auto *weight_dz = reinterpret_cast<float *>(k_param_->fil_ptr) + dz * weight_z_step;
            auto *bias_z    = reinterpret_cast<T *>(k_param_->bias) + dz;

            RunCorner(dst_z, src_z, weight_dz, 0, 0, k_param_->ow, t);
            RunCorner(dst_z, src_z, weight_dz, 0, b, k_param_->ow, k_param_->oh);
            RunCorner(dst_z, src_z, weight_dz, 0, t, l, b);
            RunCorner(dst_z, src_z, weight_dz, r, t, k_param_->ow, b);

            if (r > l && b > t) {
                dw_full(dst_z + t * k_param_->ow * 4 + l * 4,
                        src_z + (t * param->strides[1] - param->pads[2]) * k_param_->iw * 4 +
                            (l * param->strides[0] - param->pads[0]) * 4,
                        weight_dz, r - l, param->strides[0] * 4, param->kernels[0], param->kernels[1], dilate_x_step,
                        dilate_y_step, b - t, k_param_->iw * 4 * param->strides[1], k_param_->ow * 4);
            }
        }
    }

    PostExec<T>(outputs);

    return TNN_OK;
}

}  // namespace TNN_NS
