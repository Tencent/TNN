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

#include "tnn/device/x86/acc/convolution/x86_conv_layer_common.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {
/*
X86ConvLayerCommonas as the last solution, always return true
handle the case group != 1, dilate != 1, any pads and strides
*/
bool X86ConvLayerCommon::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
    return true;
}

X86ConvLayerCommon::~X86ConvLayerCommon() {}

Status X86ConvLayerCommon::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status X86ConvLayerCommon::allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status X86ConvLayerCommon::allocateBufferBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    if (!buffer_bias_.GetBytesSize()) {
        auto dims_output = outputs[0]->GetBlobDesc().dims;
        int total_byte_size = ROUND_UP(dims_output[1], 8) * DataTypeUtils::GetBytesSize(conv_res->bias_handle.GetDataType());
        RawBuffer temp_buffer(total_byte_size);
        if (conv_param->bias) {
            const int bias_handle_size    = conv_res->bias_handle.GetBytesSize();
            const float *bias_handle_data = conv_res->bias_handle.force_to<float *>();
            memcpy(temp_buffer.force_to<float *>(), conv_res->bias_handle.force_to<float *>(), bias_handle_size);
        }
        buffer_bias_ = temp_buffer;
    }
    return TNN_OK;
}

Status X86ConvLayerCommon::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = X86LayerAcc::Init(context, param, resource, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    RETURN_ON_NEQ(allocateBufferWeight(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferBias(inputs, outputs), TNN_OK);

    auto conv_param    = dynamic_cast<ConvLayerParam *>(param);
    CHECK_PARAM_NULL(conv_param);
    auto conv_resource = dynamic_cast<ConvLayerResource *>(resource);
    CHECK_PARAM_NULL(conv_resource);

    int channel    = inputs[0]->GetBlobDesc().dims[1];
    int kernel_w   = conv_param->kernels[0];
    int kernel_h   = conv_param->kernels[1];
    int group      = conv_param->group;

    if (conv_param->kernels[0] == 1 && conv_param->kernels[1] == 1 &&
        conv_param->strides[0] == 1 && conv_param->strides[1] == 1 &&
        conv_param->pads[0] == 0 && conv_param->pads[2] == 0) {
        do_im2col_ = false;
    }
    int height_out = outputs[0]->GetBlobDesc().dims[2];
    int width_out  = outputs[0]->GetBlobDesc().dims[3];

    size_t col_offset_ = kernel_h * kernel_w * height_out * width_out * (channel / group);
    col_buffer_ = RawBuffer(col_offset_ * group * sizeof(float));
    col_buffer_.SetDataType(DATA_TYPE_FLOAT);

    return TNN_OK;
}

Status X86ConvLayerCommon::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob    = inputs[0];
    Blob *output_blob   = outputs[0];
    auto input_dims     = inputs[0]->GetBlobDesc().dims;
    auto output_dims    = outputs[0]->GetBlobDesc().dims;
    void *input_ptr     = input_blob->GetHandle().base;
    void *output_ptr    = output_blob->GetHandle().base;
    auto param = dynamic_cast<ConvLayerParam *>(param_);
    auto resource = dynamic_cast<ConvLayerResource *>(resource_);

    int conv_in_offset_ = input_dims[2] * input_dims[3] * input_dims[1];
    int conv_out_spatial_dim_ = output_dims[2] * output_dims[3];
    int output_offset_ = output_dims[1] * conv_out_spatial_dim_ / param->group;
    size_t weight_offset_ = param->kernels[0] * param->kernels[1] * input_dims[1] * output_dims[1] / param->group / param->group;
    size_t col_offset_ = param->kernels[0] * param->kernels[1] * output_dims[2] * output_dims[3] * (input_dims[1] / param->group);

    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto input_data = static_cast<float*>(input_ptr);
        auto output_data = static_cast<float*>(output_ptr);
        auto weights_data = resource->filter_handle.force_to<float*>();
        float *bias_data  = buffer_bias_.force_to<float*>();
        float *col_buff;
        for (size_t b = 0; b < outputs[0]->GetBlobDesc().dims[0]; b++) {
            if (do_im2col_) {
                col_buff = col_buffer_.force_to<float*>();
                X86_IM2COL(input_data + b * conv_in_offset_, input_dims[1],
                           input_dims[2], input_dims[3],
                           param->kernels[1], param->kernels[0], 
                           param->pads[2], param->pads[0], 
                           param->strides[1], param->strides[0], 
                           param->dialations[1], param->dialations[0],
                           col_buff);
            } else {
                col_buff = input_data + b * conv_in_offset_;
            }
            for (int g = 0; g < param->group; g++) {
                X86_matrixMul(param->output_channel / param->group,
                              conv_out_spatial_dim_,
                              input_dims[1] * param->kernels[0] * param->kernels[1] / param->group,
                              weights_data + weight_offset_ * g,
                              col_buff + col_offset_ * g,
                              output_data + (b * param->group + g) * output_offset_,
                              param->bias, bias_data + g * param->output_channel / param->group, param->activation_type);
            }
        }
    } else {
        return Status(TNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT, "Error: x86 device not support this data type");
    }
    return TNN_OK;
}
}  // namespace TNN_NS
