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

#include "tnn/device/metal/acc/convolution/metal_conv_layer_common.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils_inner.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
bool MetalConvLayerCommon::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    return true;
}

MetalConvLayerCommon::~MetalConvLayerCommon() {}

Status MetalConvLayerCommon::AllocateBufferWeight(const std::vector<Blob *> &inputs,
                                                  const std::vector<Blob *> &outputs) {
    ConvLayerParam *layer_param  = dynamic_cast<ConvLayerParam *>(param_);
    ConvLayerResource *layer_res = dynamic_cast<ConvLayerResource *>(resource_);
    auto dims_input              = inputs[0]->GetBlobDesc().dims;
    auto dims_output             = outputs[0]->GetBlobDesc().dims;
    const int input_channel      = dims_input[1];
    const int output_channel     = dims_output[1];
    const int group     = layer_param->group;
    const int goc       = output_channel / group;
    const int gic       = input_channel / group;
    is_channel_4x_ = group == 1 || (group > 1 && (gic % 4 == 0) && (goc % 4 == 0));

    Status status = TNN_OK;
    if (!buffer_weight_) {
        int kw = layer_param->kernels[0];
        int kh = layer_param->kernels[1];

        if (is_channel_4x_) {
            buffer_weight_ = AllocatePackedGOIHW16MetalBufferFormRawBuffer(
                layer_res->filter_handle, {output_channel, input_channel, kh, kw}, layer_param->group, status);
        } else {
            buffer_weight_ = AllocatePackedGOIHW4MetalBufferFormRawBuffer(
                layer_res->filter_handle, {output_channel, input_channel, kh, kw}, layer_param->group, status);
        }
    }
    return status;
}

Status MetalConvLayerCommon::AllocateBufferBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<ConvLayerParam *>(param_);
    auto layer_res   = dynamic_cast<ConvLayerResource *>(resource_);

    Status status = TNN_OK;
    // buffer_bias_
    if (!buffer_bias_) {
        if (layer_param->bias) {
            auto dims_output = outputs[0]->GetBlobDesc().dims;
            const DataType data_type  = layer_res->bias_handle.GetDataType();
            bias_datatype_bytes_ = DataTypeUtils::GetBytesSize(data_type);
            buffer_bias_ = AllocateMetalBufferFormRawBuffer1D(layer_res->bias_handle, dims_output[1], status);
        } else {
            //防止bind时候为空
            buffer_bias_ = buffer_weight_;
        }
    }
    return status;
}

Status MetalConvLayerCommon::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    id<MTLDevice> device       = [TNNMetalDeviceImpl sharedDevice];
    auto conv_param = dynamic_cast<ConvLayerParam *>(param_);

    const int group  = conv_param->group;
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    const int goc    = dims_output[1] / group;
    const int gic    = dims_input[1] / group;
    // buffer_param_
    {
        MetalConvParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        SetDefaultMetalConvParams(metal_params, conv_param);
        
        if (is_channel_4x_) {
            metal_params.input_slice            = UP_DIV(dims_input[1], 4) / group;
            metal_params.input_slice_per_group  = metal_params.input_slice;
            metal_params.output_slice           = UP_DIV(dims_output[1], 4) / group;
            metal_params.output_slice_per_group = metal_params.output_slice;
        } else {
            metal_params.input_slice            = UP_DIV(gic, 4);
            metal_params.input_slice_per_group  = gic;
            metal_params.output_slice           = UP_DIV(goc, 4);
            metal_params.output_slice_per_group = goc;
        }

        metal_params.threadgroup_input_slice = metal_params.input_slice;
        //            metal_params.batch = dims_output[0];
        
        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalConvParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalConvLayerCommon::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = AllocateBufferWeight(inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    status = AllocateBufferBias(inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    status = AllocateBufferParam(inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    return TNN_OK;
}

std::string MetalConvLayerCommon::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (is_channel_4x_)
        return "convolution_common_4x";
    return "convolution_common";
}

Status MetalConvLayerCommon::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto layer_param = dynamic_cast<ConvLayerParam *>(param_);
    
    auto output = outputs[0];
    auto dims_output  = output->GetBlobDesc().dims;
    // group = 7, output_channel = 35
    auto output_slice = UP_DIV(dims_output[1], 4);
    auto output_slice_per_group = output_slice / layer_param->group;
    output_slice_per_group = output_slice_per_group > 0 ? output_slice_per_group : 1;
    if (is_channel_4x_ == false) {
        auto goc = dims_output[1] / layer_param->group;
        output_slice_per_group = UP_DIV(goc, 4);
    }
    size = MTLSizeMake(dims_output[3], dims_output[2], output_slice_per_group);
    return TNN_OK;
}

Status MetalConvLayerCommon::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<ConvLayerParam *>(param_);
    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;
    const int group  = layer_param->group;
    const int batch  = dims_output[0];
    const int goc    = dims_output[1] / group;
    const int gic    = dims_input[1] / group;
    
    auto context_impl = context_->getMetalContextImpl();
    auto encoder = [context_impl encoder];
    encoder.label = GetKernelLabel();
    
    int data_byte_size = DataTypeUtils::GetBytesSize(output->GetBlobDesc().data_type);
    
    auto input_bytes = dims_input[3] * dims_input[2] * ROUND_UP(dims_input[1], 4) * data_byte_size;
    auto input_bytes_per_group = input_bytes / group;
    auto output_bytes = dims_output[3] * dims_output[2] * ROUND_UP(dims_output[1], 4) * data_byte_size;
    auto output_bytes_per_group = output_bytes / group;
    auto bias_bytes_per_group = dims_output[1] / group * bias_datatype_bytes_;
    if (is_channel_4x_ == false) {
        // compute offset within kernel
        input_bytes_per_group = 0;
        output_bytes_per_group = 0;
    }

    Status status = TNN_OK;
    
    do {
        MTLSize threads;
        status = ComputeThreadSize(inputs, outputs, threads);
        BREAK_IF(status != TNN_OK);
        
        auto kernel_name = KernelName(inputs, outputs);
        if (kernel_name.length() <= 0) {
            status = Status(TNNERR_LAYER_ERR, "empty kernel name");
            break;
        }
        
        MetalBandwidth bandwidth;
        status = [context_impl load:[NSString stringWithUTF8String:kernel_name.c_str()]
                            encoder:encoder
                          bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
        
        for (int b = 0; b < batch; b++) {
            for (int g = 0; g < group; g++) {
                [encoder
                    setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                       offset:(NSUInteger)input->GetHandle().bytes_offset + (b * input_bytes + g * input_bytes_per_group)
                      atIndex:0];
                [encoder
                    setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                       offset:(NSUInteger)output->GetHandle().bytes_offset + (b * output_bytes + g * output_bytes_per_group)
                      atIndex:1];
                [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
                [encoder setBuffer:buffer_weight_ offset:g * buffer_weight_.length / group atIndex:3];
                // bias may be padded
                [encoder setBuffer:buffer_bias_ offset:g * bias_bytes_per_group atIndex:4];
                [encoder setBytes :&g       length:sizeof(int) atIndex:5];

                status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];

                if (status != TNN_OK) {
                    [encoder endEncoding];
                    return status;
                }
            }
        }
    } while (0);
    
    [encoder endEncoding];
    
    if (status == TNN_OK) {
        [context_impl commit];
        TNN_PRINT_ENCODER(context_, encoder, this);
    }
    return status;
}

} // namespace TNN_NS
