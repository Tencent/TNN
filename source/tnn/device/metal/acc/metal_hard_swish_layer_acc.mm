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
#include "tnn/device/metal/acc/metal_multidir_broadcast_layer_acc.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_METAL_ACC(HardSwish, LAYER_HARDSWISH);
Status MetalHardSwishLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalHardSwishLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<HardSwishLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: HardSwishLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "HardSwishLayerParam is nil");
    }

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalHardSigmoidParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);

        metal_params.broadcast_input0 = layer_param->input0_broadcast_type;
        metal_params.broadcast_input1 = layer_param->input1_broadcast_type;
        
        metal_params.alpha = layer_param->alpha;
        metal_params.beta  = layer_param->beta;
        metal_params.min   = -metal_params.beta / metal_params.alpha;
        metal_params.max   = (1.0f - metal_params.beta) / metal_params.alpha;

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalHardSigmoidParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

std::string MetalHardSwishLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "hard_swish";
}

Status MetalHardSwishLayerAcc::SetKernelEncoderParam(
                                                     id<MTLComputeCommandEncoder> encoder,
                                                     const std::vector<Blob *> &inputs,
                                                     const std::vector<Blob *> &outputs) {
    auto input0 = inputs[0];
    auto input1 = inputs.size() > 1 ? inputs[1] : input0;
    auto output = outputs[0];
    
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input0->GetHandle().base
                offset:(NSUInteger)input0->GetHandle().bytes_offset
               atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input1->GetHandle().base
                offset:(NSUInteger)input1->GetHandle().bytes_offset
               atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                offset:(NSUInteger)output->GetHandle().bytes_offset
               atIndex:2];
    [encoder setBuffer:buffer_param_ offset:0 atIndex:3];
    return TNN_OK;
}

Status MetalHardSwishLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    return MetalLayerAcc::ComputeThreadSize(inputs, outputs, size);
}

Status MetalHardSwishLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(HardSwish, LAYER_HARDSWISH);
REGISTER_METAL_LAYOUT(LAYER_HARDSWISH, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
