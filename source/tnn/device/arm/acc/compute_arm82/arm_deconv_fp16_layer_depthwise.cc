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
#include "tnn/device/arm/acc/deconvolution/arm_deconv_fp16_layer_depthwise.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
bool ArmDeconvFp16LayerDepthwise::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    auto dims_input          = inputs[0]->GetBlobDesc().dims;
    auto dims_output         = outputs[0]->GetBlobDesc().dims;
    const int input_channel  = dims_input[1];
    const int output_channel = dims_output[1];

    return param->group == input_channel && param->group == output_channel;
}

ArmDeconvFp16LayerDepthwise::~ArmDeconvFp16LayerDepthwise() {}

Status ArmDeconvFp16LayerDepthwise::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto output      = inputs[0];
    auto input       = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    const int batch    = dims_output[0];
    int src_width      = dims_input[3];
    int src_height     = dims_input[2];
    int dst_width      = dims_output[3];
    int dst_height     = dims_output[2];
    int dst_depth_div8 = UP_DIV(dims_output[1], 8);
    int dst_z_step     = dst_width * dst_height * 8;
    int src_z_step     = src_width * src_height * 8;
    int dst_y_step     = dst_width * 8;
    int src_y_step     = src_width * 8;
    int kernel_x       = conv_param->kernels[0];
    int kernel_y       = conv_param->kernels[1];
    int stride_x       = conv_param->strides[0];
    int stride_y       = conv_param->strides[1];
    int pad_x          = conv_param->pads[0];
    int pad_y          = conv_param->pads[2];
    int dilate_x       = conv_param->dialations[0];
    int dilate_y       = conv_param->dialations[1];
    int dilate_y_step  = src_width * 8 * dilate_y;
    int dilate_x_step  = 8 * dilate_x;
    int weight_z_step  = kernel_x * kernel_y * 8;

    int l = 0, t = 0, r = dst_width, b = dst_height;
    for (; l * stride_x - pad_x < 0; l++) {
        // do nothing
    }
    for (; t * stride_y - pad_y < 0; t++) {
        // do nothing
    }
    for (; (r - 1) * stride_x - pad_x + kernel_x * dilate_x > src_width && r > l; r--) {
        // do nothing
    }
    for (; (b - 1) * stride_y - pad_y + kernel_y * dilate_y > src_height && b > t; b--) {
        // do nothing
    }
    auto RunCorner = [=](fp16_t *dst_z, fp16_t *src_z, const fp16_t *weight_dz, int Left, int Top, int Right, int Bot) {
        for (int dy = Top; dy < Bot; ++dy) {
            fp16_t *dst_y  = dst_z + dy * dst_y_step;
            int srcStartY  = dy * stride_y - pad_y;
            fp16_t *src_dy = src_z + srcStartY * src_y_step;
            int sfy        = MAX(0, (UP_DIV(-srcStartY, dilate_y)));
            int efy        = MIN(kernel_y, UP_DIV(src_height - srcStartY, dilate_y));
            for (int dx = Left; dx < Right; ++dx) {
                fp16_t *dst_x  = dst_y + 8 * dx;
                int srcStartX  = dx * stride_x - pad_x;
                fp16_t *src_dx = src_dy + srcStartX * 8;
                int sfx        = MAX(0, (UP_DIV(-srcStartX, dilate_x)));
                int efx        = MIN(kernel_x, UP_DIV(src_width - srcStartX, dilate_x));
                DepthwiseUnitDeconv(dst_x, src_dx + (sfx * dilate_x + sfy * dilate_y * src_width) * 8,
                                    weight_dz + 8 * (kernel_x * sfy + sfx), efx - sfx, efy - sfy, 8 * kernel_x,
                                    dilate_x_step, dilate_y_step);
            }
        }
    };

    fp16_t *src_orign = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input->GetHandle()));
    fp16_t *dst_orign = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto src_ptr = src_orign + batch_idx * src_width * src_height * ROUND_UP(dims_input[1], 8);
        auto dst_ptr = dst_orign + batch_idx * dst_width * dst_height * ROUND_UP(dims_output[1], 8);

        memset(src_ptr, 0, src_width * src_height * dst_depth_div8 * 8 * sizeof(fp16_t));

        for (int dz = 0; dz < dst_depth_div8; dz++) {
            fp16_t *dst_z     = dst_ptr + dst_z_step * dz;
            fp16_t *src_z     = src_ptr + src_z_step * dz;
            fp16_t *weight_dz = buffer_weight_.force_to<fp16_t *>() + dz * weight_z_step;

            RunCorner(dst_z, src_z, weight_dz, 0, 0, dst_width, t);
            RunCorner(dst_z, src_z, weight_dz, 0, b, dst_width, dst_height);
            RunCorner(dst_z, src_z, weight_dz, 0, t, l, b);
            RunCorner(dst_z, src_z, weight_dz, r, t, dst_width, b);

            if (r > l) {
                for (int dy = t; dy < b; dy++) {
                    fp16_t *dst_y  = dst_z + dy * dst_y_step;
                    int srcStartY  = dy * stride_y - pad_y;
                    fp16_t *src_dy = src_z + srcStartY * src_y_step;
                    DepthwiseDeconv(dst_y + l * 8, src_dy + (l * stride_x - pad_x) * 8, weight_dz, r - l, stride_x * 8,
                                    kernel_x, kernel_y, dilate_x_step, dilate_y_step);
                }
            }
        }
    }

    PostExec<fp16_t>(outputs);

    return TNN_OK;
}

}  // namespace TNN_NS
#endif
