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

#include "tnn/device/metal/acc/convolution/metal_conv_layer_1x1.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {
bool MetalConvLayer1x1::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    auto kernel_x = param->kernels[0], kernel_y = param->kernels[1];
    auto dilate_x = param->dialations[0], dilate_y = param->dialations[1];
    auto stride_x = param->strides[0], stride_y = param->strides[1];
    auto pad_x = param->pads[0], pad_y = param->pads[0];
    return kernel_x == 1 && kernel_y == 1 && dilate_x == 1 && dilate_y == 1 && pad_x == 0 && pad_y == 0 &&
           stride_x == 1 && stride_y == 1;
}

MetalConvLayer1x1::~MetalConvLayer1x1() {}

Status MetalConvLayer1x1::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    id<MTLDevice> device       = [TNNMetalDeviceImpl sharedDevice];
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    const int group  = conv_param->group;
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    const int goc = dims_output[1] / group;
    const int gic = dims_input[1] / group;

    if (group > 1 && (gic % 4 != 0) && (goc % 4 != 0)) {
        LOGD("convolution: channel per group must be 4x\n");
        return Status(TNNERR_LAYER_ERR, "convolution: channel per group must be 4x");
    }
    
    // buffer_param_
    {
        MetalConvParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        SetDefaultMetalConvParams(metal_params, conv_param);
        
        metal_params.input_slice_per_group  = metal_params.input_slice / group;
        metal_params.output_slice_per_group = metal_params.output_slice / group;
        
        

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalConvParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

std::string MetalConvLayer1x1::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "convolution_1x1";
}

Status MetalConvLayer1x1::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
    [encoder setBuffer:buffer_weight_
                offset:0
               atIndex:3];
    [encoder setBuffer:buffer_bias_
                offset:0
               atIndex:4];
    return TNN_OK;
}

Status MetalConvLayer1x1::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto layer_param = dynamic_cast<ConvLayerParam *>(param_);
    
    auto dims_output  = outputs[0]->GetBlobDesc().dims;
    auto output_slice = UP_DIV(dims_output[1], 4);
    size = MTLSizeMake(dims_output[3]*dims_output[2], output_slice, dims_output[0]);
    return TNN_OK;
}

Status MetalConvLayer1x1::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

} // namespace TNN_NS
