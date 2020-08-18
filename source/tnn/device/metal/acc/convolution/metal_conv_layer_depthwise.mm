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

#include "tnn/device/metal/acc/convolution/metal_conv_layer_depthwise.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils.h"

namespace TNN_NS {
bool MetalConvLayerDepthwise::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    auto input_dims          = inputs[0]->GetBlobDesc().dims;
    auto output_dims         = outputs[0]->GetBlobDesc().dims;

    return param->group == input_dims[1] && param->group == output_dims[1];
}

MetalConvLayerDepthwise::~MetalConvLayerDepthwise() {}

Status MetalConvLayerDepthwise::AllocateBufferWeight(const std::vector<Blob *> &inputs,
                                                     const std::vector<Blob *> &outputs) {
    auto param  = dynamic_cast<ConvLayerParam *>(param_);
    auto resource = dynamic_cast<ConvLayerResource *>(resource_);

    Status status = TNN_OK;
    if (!buffer_weight_) {
        int kw = param->kernels[0];
        int kh = param->kernels[1];

        const int group = param->group;
        buffer_weight_ =  AllocatePackedNC4HW4MetalBufferFormRawBuffer(
                                                                       resource->filter_handle,
                                                                       {1, group, kh, kw},
                                                                       group,
                                                                       status);
    }
    return status;
}

Status MetalConvLayerDepthwise::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                    const std::vector<Blob *> &outputs) {
    id<MTLDevice> device        = [TNNMetalDeviceImpl sharedDevice];
    ConvLayerParam *layer_param = dynamic_cast<ConvLayerParam *>(param_);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalConvParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        SetDefaultMetalConvParams(metal_params, layer_param);

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalConvParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    return TNN_OK;
}

std::string MetalConvLayerDepthwise::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "convolution_depthwise";
}

Status MetalConvLayerDepthwise::SetKernelEncoderParam(
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

Status MetalConvLayerDepthwise::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto output = outputs[0];
    auto dims_output  = output->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, false);
    return TNN_OK;
}

Status MetalConvLayerDepthwise::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

} // namespace TNN_NS
