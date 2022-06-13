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

#include "tnn/device/x86/acc/convolution/x86_conv_int8_layer_depthwise.h"

#include "tnn/device/x86/x86_common.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/device/x86/acc/compute/x86_compute_int8.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
bool X86ConvInt8LayerDepthwise::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                           const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }
    if (inputs[0]->GetBlobDesc().data_type != DATA_TYPE_INT8) {
        return false;
    }
    if (param->fusion_type != FusionType_None) {
        return false;
    }

    auto dims_input          = inputs[0]->GetBlobDesc().dims;
    auto dims_output         = outputs[0]->GetBlobDesc().dims;
    const int input_channel  = dims_input[1];
    const int output_channel = dims_output[1];

    return param->group == input_channel && param->group == output_channel;
}

X86ConvInt8LayerDepthwise::~X86ConvInt8LayerDepthwise() {}

Status X86ConvInt8LayerDepthwise::allocateBufferWeight(const std::vector<Blob *> &inputs,
                                                      const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    if (!buffer_weight_.GetBytesSize()) {
        int8_t *filter = conv_res->filter_handle.force_to<int8_t *>();
        CHECK_PARAM_NULL(filter);
        int kw             = conv_param->kernels[0];
        int kh             = conv_param->kernels[1];
        const int channel  = inputs[0]->GetBlobDesc().dims[1];
        const int c_4      = ROUND_UP(channel, 4);
        int data_byte_size = c_4 * kh * kw;
        RawBuffer temp_buffer(data_byte_size);
        int8_t *temp_ptr = temp_buffer.force_to<int8_t *>();

        for (int c = 0; c < channel; c++) {
            int8_t *f_c = filter + c * kw * kh;
            int8_t *t_c = temp_ptr + c;
            for (int k = 0; k < kh * kw; k++) {
                t_c[k * c_4] = f_c[k];
            }
        }

        buffer_weight_ = temp_buffer;
    }
    return TNN_OK;
}

Status X86ConvInt8LayerDepthwise::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    const int batch  = dims_output[0];
    const int group  = conv_param->group;
    auto input_width = dims_input[3], input_height = dims_input[2], ic = dims_input[1];
    auto output_width = dims_output[3], output_height = dims_output[2], oc = dims_output[1];
    auto ic_r4 = ROUND_UP(dims_input[1], 4);
    auto oc_r4 = ROUND_UP(dims_output[1], 4);

    int kernel_x = conv_param->kernels[0];
    int kernel_y = conv_param->kernels[1];
    int stride_x = conv_param->strides[0];
    int stride_y = conv_param->strides[1];
    int pad_x    = conv_param->pads[0];
    int pad_y    = conv_param->pads[2];
    int dilate_x = conv_param->dialations[0];
    int dilate_y = conv_param->dialations[1];

    const int dst_y_step = output_width * oc_r4;
    const int src_y_step = input_width * ic_r4;

    int8_t *input_data  = handle_ptr<int8_t *>(input->GetHandle());
    int8_t *output_data = handle_ptr<int8_t *>(output->GetHandle());

    const int32_t *bias_data = buffer_bias_.force_to<int32_t *>();
    const float *scale_data  = buffer_scale_.force_to<float *>();
    int8_t *weight_data      = buffer_weight_.force_to<int8_t *>();

    int l = 0, t = 0, r = output_width, b = output_height;
    for (; l * stride_x - pad_x < 0; l++)
        ;
    for (; t * stride_y - pad_y < 0; t++)
        ;
    for (; (r - 1) * stride_x - pad_x + kernel_x * dilate_x > input_width && r > l; r--)
        ;
    for (; (b - 1) * stride_y - pad_y + kernel_y * dilate_y > input_height && b > t; b--)
        ;

    auto RunCorner = [=](int8_t *dst_z, const int8_t *src_z, int left, int top, int right, int bottom) {
        for (long dy = top; dy < bottom; ++dy) {
            auto dst_y             = dst_z + dy * dst_y_step;
            const long src_start_y = dy * stride_y - pad_y;
            const auto src_y       = src_z + src_start_y * src_y_step;
            const long sfy         = MAX(0, (UP_DIV(-src_start_y, dilate_y)));
            const long efy         = MIN(kernel_y, (UP_DIV(dims_input[2] - src_start_y, dilate_y)));
            for (long dx = left; dx < right; ++dx) {
                auto dst_x             = dst_y + oc_r4 * dx;
                const long src_start_x = dx * stride_x - pad_x;
                const auto src_x       = src_y + src_start_x * oc_r4;
                const long sfx         = MAX(0, (UP_DIV(-src_start_x, dilate_x)));
                const long efx         = MIN(kernel_x, (UP_DIV(dims_input[3] - src_start_x, dilate_x)));
                const long srcIndex    = (sfx * dilate_x + sfy * dilate_y * dims_input[3]) * oc_r4;
                const long weightIndex = (kernel_x * sfy + sfx) * oc_r4;

                X86DepthwiseI8Unit(dst_x, src_x + srcIndex, weight_data + weightIndex, bias_data,
                                efx - sfx, efy - sfy, oc_r4 * kernel_x, src_y_step * dilate_y,
                                oc_r4 * dilate_x, scale_data, oc_r4);
            }
        }
    };

    for (int bIndex = 0; bIndex < batch; ++bIndex) {
        const auto input_batch = input_data + bIndex * src_y_step * input_height;
        auto output_batch      = output_data + bIndex * dst_y_step * output_height;

        long src_w_step = oc_r4 * conv_param->strides[0];
        auto dwfunc     = X86DepthwiseI8General;

        if (kernel_x == kernel_y && kernel_x == 3 && oc_r4 >= 8 && dilate_x == 1 && dilate_y == 1) {
            dwfunc = X86DepthwiseI8K3;
        } else if (kernel_x == kernel_y && kernel_x == 5 && oc_r4 >= 8 && dilate_x == 1 && dilate_y == 1) {
            dwfunc = X86DepthwiseI8K5;
        }

        OMP_PARALLEL_SECTIONS_ {
            OMP_SECTION_ {
                // top corner
                RunCorner(output_batch, input_batch, 0, 0, dims_output[3], t);
            }
            OMP_SECTION_ {
                // bottom corner
                RunCorner(output_batch, input_batch, 0, b, dims_output[3], dims_output[2]);
            }
            OMP_SECTION_ {
                // left corner
                RunCorner(output_batch, input_batch, 0, t, l, b);
            }
            OMP_SECTION_ {
                // bottom corner
                RunCorner(output_batch, input_batch, r, t, dims_output[3], b);
            }
        }
        if (r > l && b > t) {
            OMP_PARALLEL_FOR_GUIDED_
            for (long dy = t; dy < b; ++dy) {
                const long src_start_y = dy * conv_param->strides[1] - conv_param->pads[2];
                const auto src_dy      = input_batch + src_start_y * src_y_step;
                auto dst_y             = output_batch + dy * dst_y_step;
                dwfunc(dst_y + l * oc_r4,
                       src_dy + (l * conv_param->strides[0] - conv_param->pads[0]) * oc_r4,
                       weight_data, bias_data,
                       r - l, src_y_step * dilate_y, oc_r4 * dilate_x, src_w_step, oc_r4,
                       conv_param->kernels[0], conv_param->kernels[1], scale_data);
            }
        }

        if (conv_param->activation_type == ActivationType_ReLU) {
            X86ReluInt8(output_batch, output_batch, output_height * dst_y_step);
        } else if (conv_param->activation_type == ActivationType_ReLU6) {
            X86Relu6Int8(output_batch, output_batch, relu6_max_.force_to<int8_t *>(), output_height * output_width, oc_r4);
        }
    }
    return TNN_OK;
}

}  // namespace TNN_NS
