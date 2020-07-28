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

#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_depthwise.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
bool ArmConvInt8LayerDepthwise::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                           const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }
    if (inputs[0]->GetBlobDesc().data_type != DATA_TYPE_INT8) {
        return false;
    }

    auto dims_input          = inputs[0]->GetBlobDesc().dims;
    auto dims_output         = outputs[0]->GetBlobDesc().dims;
    const int input_channel  = dims_input[1];
    const int output_channel = dims_output[1];

    return param->group == input_channel && param->group == output_channel;
}

ArmConvInt8LayerDepthwise::~ArmConvInt8LayerDepthwise() {}

Status ArmConvInt8LayerDepthwise::allocateBufferParam(const std::vector<Blob *> &inputs,
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

Status ArmConvInt8LayerDepthwise::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
    auto ic_4 = UP_DIV(dims_input[1], 4);
    auto oc_4 = UP_DIV(dims_output[1], 4);

    int kernel_x = conv_param->kernels[0];
    int kernel_y = conv_param->kernels[1];
    int stride_x = conv_param->strides[0];
    int stride_y = conv_param->strides[1];
    int pad_x    = conv_param->pads[0];
    int pad_y    = conv_param->pads[2];
    int dilate_x = conv_param->dialations[0];
    int dilate_y = conv_param->dialations[1];

    const int dst_y_step = output_width * oc_4 * 4;
    const int src_y_step = input_width * ic_4 * 4;

    int8_t *input_data  = reinterpret_cast<int8_t *>(GetBlobHandlePtr(input->GetHandle()));
    int8_t *output_data = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));

    const int32_t *bias_data = buffer_bias_.force_to<int32_t *>();
    const float *scale_data  = buffer_scale_.force_to<float *>();
    int8_t *weight_data      = buffer_weight_.force_to<int8_t *>();

    int l = 0, t = 0, r = output_width, b = output_height;
    for (; l * stride_x - pad_x < 0; l++)
        ;
    for (; t * stride_y - pad_y < 0; t++)
        ;
    for (; (r - 1) * stride_x - pad_x + kernel_x > input_width && r > l; r--)
        ;
    for (; (b - 1) * stride_y - pad_y + kernel_y > input_height && b > t; b--)
        ;

    for (int bIndex = 0; bIndex < batch; ++bIndex) {
        const auto input_batch = input_data + bIndex * src_y_step * input_height;
        auto output_batch      = output_data + bIndex * dst_y_step * output_height;

        DepthwiseConvI8(input_batch, output_batch, oc_4 * 4, src_y_step, dst_y_step, output_height, output_width,
                        input_height, input_width, l, r, t, b, kernel_x, weight_data, bias_data, scale_data, stride_x,
                        pad_x, k_param_.get());
        if (conv_param->activation_type == ActivationType_ReLU) {
            ReluInt8(output_batch, output_batch, output_height * dst_y_step);
        }
    }
    return TNN_OK;
}

}  // namespace TNN_NS
