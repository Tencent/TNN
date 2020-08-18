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
#include "tnn/device/metal/acc/deconvolution/metal_deconv_layer_depthwise.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils.h"

namespace TNN_NS {
bool MetalDeconvLayerDepthwise::isPrefered(ConvLayerParam *param,
                                           const std::vector<Blob *> &inputs,
                                           const std::vector<Blob *> &outputs) {
    return MetalConvLayerDepthwise::isPrefered(param, inputs, outputs);
}

MetalDeconvLayerDepthwise::~MetalDeconvLayerDepthwise() {}

Status MetalDeconvLayerDepthwise::AllocateBufferWeight(
    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *layer_param = dynamic_cast<ConvLayerParam *>(param_);
    ConvLayerResource *layer_res =
        dynamic_cast<ConvLayerResource *>(resource_);

    Status status = TNN_OK;
    if (!buffer_weight_) {
        int kw = layer_param->kernels[0];
        int kh = layer_param->kernels[1];

        const int group  = layer_param->group;
        buffer_weight_ = AllocatePackedNC4HW4MetalBufferFormRawBuffer(layer_res->filter_handle,
                                                                {1, group, kh, kw},
                                                                group, status);
    }
    return status;
}

std::string MetalDeconvLayerDepthwise::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "deconv_depthwise";
}

Status MetalDeconvLayerDepthwise::ComputeThreadSize(
                                                    const std::vector<Blob *> &inputs,
                                                    const std::vector<Blob *> &outputs,
                                                    MTLSize &size) {
    auto output = outputs[0];
    auto dims_output  = output->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, false);
    return TNN_OK;
}

Status MetalDeconvLayerDepthwise::SetKernelEncoderParam(
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

Status MetalDeconvLayerDepthwise::Forward(const std::vector<Blob *> &inputs,
                                          const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<ConvLayerParam *>(param_);
    auto input                   = inputs[0];
    auto dims_input              = input->GetBlobDesc().dims;
    if (!layer_param || dims_input[0] != 1 || layer_param->group != dims_input[1]) {
        LOGE("Error: batch size or group is not support\n");
        return Status(TNNERR_LAYER_ERR, "batch size or group is not support");
    }
    
    return MetalLayerAcc::Forward(inputs, outputs);
}

} // namespace TNN_NS
