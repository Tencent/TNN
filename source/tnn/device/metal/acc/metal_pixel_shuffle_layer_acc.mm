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

#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/acc/metal_layer_acc.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_METAL_ACC(PixelShuffle, LAYER_PIXEL_SHUFFLE);

Status MetalPixelShuffleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalPixelShuffleLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto layer_param     = dynamic_cast<PixelShuffleLayerParam *>(param_);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalPixelShuffleParams metal_params;
        metal_params.batch          = dims_input[0];
        metal_params.input_channel  = dims_input[1];
        metal_params.input_slice    = UP_DIV(dims_input[1], 4);
        metal_params.input_height   = DimsFunctionUtils::GetDim(dims_input, 2);
        metal_params.input_width    = DimsFunctionUtils::GetDim(dims_input, 3);

        metal_params.output_slice   = UP_DIV(dims_output[1], 4);
        metal_params.output_height  = DimsFunctionUtils::GetDim(dims_output, 2);
        metal_params.output_width   = DimsFunctionUtils::GetDim(dims_output, 3);

        metal_params.upscale_factor = layer_param->upscale_factor;

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalPixelShuffleParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalPixelShuffleLayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

std::string MetalPixelShuffleLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "pixel_shuffle_common";
}

Status MetalPixelShuffleLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, false);
    return TNN_OK;
}

Status MetalPixelShuffleLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(PixelShuffle, LAYER_PIXEL_SHUFFLE);
REGISTER_METAL_LAYOUT(LAYER_PIXEL_SHUFFLE, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
