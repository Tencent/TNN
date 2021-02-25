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

namespace TNN_NS {

DECLARE_METAL_ACC(Squeeze, LAYER_SQUEEZE);

Status MetalSqueezeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}
    
Status MetalSqueezeLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto dims_input      = inputs[0]->GetBlobDesc().dims;
    auto dims_output     = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalPermuteParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        metal_params.input_batch = dims_input[0];
        buffer_param_     = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(metal_params)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}
    
Status MetalSqueezeLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs,
                                            MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, false);
    return TNN_OK;
}

std::string MetalSqueezeLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "permute_copy";
}

Status MetalSqueezeLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

Status MetalSqueezeLayerAcc::SetKernelEncoderParam(id<MTLComputeCommandEncoder> encoder,
                                                const std::vector<Blob *> &inputs,
                                                const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

REGISTER_METAL_ACC(Squeeze, LAYER_SQUEEZE);
REGISTER_METAL_LAYOUT(LAYER_SQUEEZE, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS

