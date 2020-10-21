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

#include "x86_conv_layer_acc.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include <iostream>

namespace TNN_NS {

Status X86ConvLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                             const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = X86LayerAcc::Init(context, param, resource, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    
    auto conv_param    = dynamic_cast<ConvLayerParam *>(param);
    CHECK_PARAM_NULL(conv_param);
    auto conv_resource = dynamic_cast<ConvLayerResource *>(resource);
    CHECK_PARAM_NULL(conv_resource);

    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
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

        conv_in_height_ = inputs[0]->GetBlobDesc().dims[2];
        conv_in_width_  = inputs[0]->GetBlobDesc().dims[3];
        conv_in_offset_ = conv_in_height_ * conv_in_width_ * channel;
        conv_in_channels_ = inputs[0]->GetBlobDesc().dims[1];
        conv_out_spatial_dim_ = height_out * width_out;
        conv_out_channles_ = outputs[0]->GetBlobDesc().dims[1];
        output_offset_ = conv_out_channles_ * conv_out_spatial_dim_ / group;
        
        weight_offset_ = kernel_h * kernel_w * channel * conv_out_channles_ / group / group; // w h
        col_offset_    = kernel_h * kernel_w * height_out * width_out * (channel / group);
        col_buffer_  = RawBuffer(col_offset_ * group * sizeof(float));
        col_buffer_.SetDataType(DATA_TYPE_FLOAT);
    } else {
        return Status(TNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT, "Error: x86 device not support this data type");
    }
    return TNN_OK;
}

Status X86ConvLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status X86ConvLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob    = inputs[0];
    Blob *output_blob   = outputs[0];
    void *input_ptr     = input_blob->GetHandle().base;
    void *output_ptr    = output_blob->GetHandle().base;
    auto param = dynamic_cast<ConvLayerParam *>(param_);
    auto resource = dynamic_cast<ConvLayerResource *>(resource_);

    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto input_data = static_cast<float*>(input_ptr);
        auto output_data = static_cast<float*>(output_ptr);
        auto weights_data = resource->filter_handle.force_to<float*>();
        float *bias_data = nullptr;
        if (resource->bias_handle.GetDataCount() != 0) {
            bias_data = resource->bias_handle.force_to<float*>();
        }
        float *col_buff;
        for (size_t b = 0; b < outputs[0]->GetBlobDesc().dims[0]; b++) {
            if (do_im2col_) {
                col_buff = col_buffer_.force_to<float*>();
                X86_IM2COL(input_data + b * conv_in_offset_, conv_in_channels_,
                           conv_in_height_, conv_in_width_,
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
                              conv_in_channels_ * param->kernels[0] * param->kernels[1] / param->group,
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

REGISTER_X86_ACC(Conv, LAYER_CONVOLUTION);

}   // namespace TNN_NS