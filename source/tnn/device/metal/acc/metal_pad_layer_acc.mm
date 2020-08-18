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

DECLARE_METAL_ACC(Pad, LAYER_PAD);

Status MetalPadLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalPadLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto layer_param     = dynamic_cast<PadLayerParam *>(param_);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalPadParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);

        metal_params.pad_l = layer_param->pads[0];
        metal_params.pad_r = layer_param->pads[1];
        metal_params.pad_t = layer_param->pads[2];
        metal_params.pad_b = layer_param->pads[3];
        metal_params.value = 0;

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalPadParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalPadLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, false);
    return TNN_OK;
}

std::string MetalPadLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PadLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return "";
    }
    
    if (layer_param->type == 1) {
        return "pad_reflect_common";
    } else if (layer_param->type == 0) {
        return "pad_const_common";
    } else {
        LOGE("Error: layer param is not supported: type:%d\n", layer_param->type);
        return "";
    }
}

Status MetalPadLayerAcc::SetKernelEncoderParam(
                                               id<MTLComputeCommandEncoder> encoder,
                                               const std::vector<Blob *> &inputs,
                                               const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

Status MetalPadLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(Pad, LAYER_PAD);

} // namespace TNN_NS
