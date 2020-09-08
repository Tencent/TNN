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

#include "tnn/device/metal/acc/deconvolution/metal_deconv_layer_common.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils.h"

namespace TNN_NS {

static int LeastCommonMultiple(int m, int n) {
    int a = m, b = n;
    while (a != b) {
        if (a > b) {
            a = a - b;
        } else {
            b = b - a;
        }
    }
    return m * n / a;
}

bool MetalDeconvLayerCommon::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs) {
    return true;
}

MetalDeconvLayerCommon::~MetalDeconvLayerCommon() {}

Status MetalDeconvLayerCommon::AllocateBufferWeight(const std::vector<Blob *> &inputs,
                                                    const std::vector<Blob *> &outputs) {
    ConvLayerParam *layer_param  = dynamic_cast<ConvLayerParam *>(param_);
    ConvLayerResource *layer_res = dynamic_cast<ConvLayerResource *>(resource_);
    auto dims_input              = inputs[0]->GetBlobDesc().dims;
    auto dims_output             = outputs[0]->GetBlobDesc().dims;
    const int input_channel      = dims_input[1];
    const int output_channel     = dims_output[1];

    Status status = TNN_OK;
    if (!buffer_weight_) {
        int kw = layer_param->kernels[0];
        int kh = layer_param->kernels[1];

        buffer_weight_ = AllocatePackedGOIHW16MetalBufferFormRawBuffer(
            layer_res->filter_handle, {output_channel, input_channel, kh, kw}, layer_param->group, status, true);
    }
    return status;
}

Status MetalDeconvLayerCommon::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    id<MTLDevice> device         = [TNNMetalDeviceImpl sharedDevice];
    ConvLayerParam *deconv_param = dynamic_cast<ConvLayerParam *>(param_);

    const int group  = deconv_param->group;
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalConvParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        SetDefaultMetalConvParams(metal_params, deconv_param);

        auto input_slice_per_group = UP_DIV(dims_input[1], 4) / group;
        input_slice_per_group      = input_slice_per_group > 0 ? input_slice_per_group : 1;
        metal_params.input_slice   = input_slice_per_group;

        auto output_slice_per_group = UP_DIV(dims_output[1], 4) / group;
        output_slice_per_group      = output_slice_per_group > 0 ? output_slice_per_group : 1;
        metal_params.output_slice   = output_slice_per_group;

        metal_params.threadgroup_input_slice = metal_params.input_slice;
        //            metal_params.batch = dims_output[0];
        
        metal_params.kernel_delta_y =
            LeastCommonMultiple(metal_params.dilation_y, metal_params.stride_y) / metal_params.dilation_y;
        metal_params.kernel_delta_x =
            LeastCommonMultiple(metal_params.dilation_x, metal_params.stride_x) / metal_params.dilation_x;
        metal_params.input_delta_y = metal_params.kernel_delta_y * metal_params.dilation_y / metal_params.stride_y;
        metal_params.input_delta_x = metal_params.kernel_delta_x * metal_params.dilation_x / metal_params.stride_x;

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalConvParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalDeconvLayerCommon::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *deconv_param = dynamic_cast<ConvLayerParam *>(param_);
    auto group                   = deconv_param->group;
    auto input                   = inputs[0];
    auto output                  = outputs[0];
    auto dims_input              = input->GetBlobDesc().dims;
    auto dims_output             = output->GetBlobDesc().dims;
    auto context_impl            = context_->getMetalContextImpl();

    if (dims_input[0] != 1) {
        LOGE("Error: batch size or group is not support\n");
        return Status(TNNERR_LAYER_ERR, "batch size or group is not support");
    }

    auto encoder = [context_impl encoder];
    encoder.label = GetKernelLabel();

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    auto input_width             = dims_input[3];
    auto input_height            = dims_input[2];
    auto input_channel           = dims_input[1];
    auto input_channel_per_group = input_channel / group;
    auto input_slice             = UP_DIV(dims_input[1], 4);
    auto input_bytes             = input_width * input_height * input_slice * 4 * data_byte_size;
    auto input_bytes_per_group   = input_bytes / group;

    auto input_slice_per_group = input_slice / group;
    input_slice_per_group      = input_slice_per_group > 0 ? input_slice_per_group : 1;

    auto output_width             = dims_output[3];
    auto output_height            = dims_output[2];
    auto output_channel           = dims_output[1];
    auto output_channel_per_group = output_channel / group;
    auto output_slice             = UP_DIV(dims_output[1], 4);
    auto output_bytes             = output_width * output_height * output_slice * 4 * data_byte_size;
    auto output_bytes_per_group   = output_bytes / group;

    auto output_slice_per_group = output_slice / group;
    output_slice_per_group      = output_slice_per_group > 0 ? output_slice_per_group : 1;

    auto kernel_size = deconv_param->kernels[0] * deconv_param->kernels[1];
    auto weight_bytes_per_group = output_slice_per_group * input_slice_per_group * kernel_size * 16 * data_byte_size;
    auto bias_bytes_per_group   = output_slice_per_group * 4 * data_byte_size;

    Status status = TNN_OK;
    MetalBandwidth bandwidth;
    MTLSize threads = {(NSUInteger)output_width, (NSUInteger)output_height, (NSUInteger)output_slice_per_group};

    bool supported = false;
    if (group == 1 || (group != 1 && (output_channel_per_group % 4) == 0 && (input_channel_per_group % 4) == 0)) {
        supported = true;
        status    = [context_impl load:@"deconv_common_group_channel_in4x_out4x" encoder:encoder bandwidth:bandwidth];
    } else {
        //注意此处先特殊处理
        if (group == 2 && output_channel_per_group == 1 && input_channel_per_group == 2) {
            supported              = true;
            status                 = [context_impl load:@"deconv_common_group_channel_in2_out1_group2"
                                encoder:encoder
                              bandwidth:bandwidth];
            input_bytes_per_group  = input_channel_per_group * output_height * output_width * data_byte_size;
            output_slice_per_group = output_channel_per_group * output_height * output_width * data_byte_size;
        } else {
            supported = false;
        }
    }
    if (!supported) {
        [encoder endEncoding];
        LOGE("Error: deconv group != 1 is not supported\n");
        return Status(TNNERR_LAYER_ERR, "deconv group != 1 is not supported");
    }

    if (status != TNN_OK) {
        [encoder endEncoding];
        return status;
    }

    if (!(group == 2 && output_channel_per_group == 1 && input_channel_per_group == 2)) {
        for (int g = 0; g < group; g++) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                        offset:(NSUInteger)input->GetHandle().bytes_offset + g * input_bytes_per_group
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                        offset:(NSUInteger)output->GetHandle().bytes_offset + g * output_bytes_per_group
                       atIndex:1];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
            [encoder setBuffer:buffer_weight_ offset:g * weight_bytes_per_group atIndex:3];
            [encoder setBuffer:buffer_bias_ offset:g * bias_bytes_per_group atIndex:4];

            status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];

            if (status != TNN_OK) {
                [encoder endEncoding];
                return status;
            }
        }
    } else {
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                    offset:(NSUInteger)input->GetHandle().bytes_offset
                   atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                    offset:(NSUInteger)output->GetHandle().bytes_offset
                   atIndex:1];
        [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
        [encoder setBuffer:buffer_weight_ offset:0 atIndex:3];
        [encoder setBuffer:buffer_bias_ offset:0 atIndex:4];

        status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];

        if (status != TNN_OK) {
            [encoder endEncoding];
            return status;
        }
    }
    [encoder endEncoding];
    [context_impl commit];
    TNN_PRINT_ENCODER(context_, encoder, this);
    return TNN_OK;
}

} // namespace TNN_NS
