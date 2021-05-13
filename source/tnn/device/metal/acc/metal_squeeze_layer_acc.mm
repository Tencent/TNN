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

DECLARE_METAL_ACC_WITH_EXTRA(Squeeze, LAYER_SQUEEZE,
    public:  virtual Status UpdateBlobDataType(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    private: bool need_reformat_ = false);

Status MetalSqueezeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalSqueezeLayerAcc::UpdateBlobDataType(const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs) {
    outputs[0]->GetBlobDesc().data_type = inputs[0]->GetBlobDesc().data_type;
    return TNN_OK;
}
    
Status MetalSqueezeLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto dims_input      = inputs[0]->GetBlobDesc().dims;
    auto dims_output     = outputs[0]->GetBlobDesc().dims;
    auto layer_param        = dynamic_cast<SqueezeLayerParam*>(param_);
    need_reformat_ = false;
    auto axes = layer_param->axes;
    for (auto axis : axes) {
        axis = axis < 0 ? axis + dims_output.size() : axis;
        need_reformat_ = need_reformat_ || axis == 0 || axis == 1;
    }
    // buffer_param_
    {
        if (need_reformat_) {
            MetalSqueezeParams metal_params;
            SetDefaultMetalParams(metal_params, dims_input, dims_output);
            FixDefaultMetalParams(metal_params, dims_input, dims_output);
            metal_params.input_channel  = dims_input[1];
            metal_params.output_channel = dims_output[1];
            metal_params.input_batch = dims_input[0];
            buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                                    length:sizeof(metal_params)
                                                   options:MTLResourceCPUCacheModeWriteCombined];
        } else {
            MetalPermuteParams metal_params;
            SetDefaultMetalParams(metal_params, dims_input, dims_output);
            FixDefaultMetalParams(metal_params, dims_input, dims_output);
            metal_params.input_batch = dims_input[0];
            buffer_param_     = [device newBufferWithBytes:(const void *)(&metal_params)
                                                    length:sizeof(metal_params)
                                                   options:MTLResourceCPUCacheModeWriteCombined];
        }
    }
    return TNN_OK;
}
    
Status MetalSqueezeLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs,
                                            MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = GetDefaultThreadSizeFusedLast(dims_output, need_reformat_);
    return TNN_OK;
}

std::string MetalSqueezeLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    const auto data_type = outputs[0]->GetBlobDesc().data_type;
    if (need_reformat_)
        return DataTypeUtils::GetBytesSize(data_type)==4? "squeeze_common_int4" : "squeeze_common";
    return DataTypeUtils::GetBytesSize(data_type)==4? "permute_copy_int4" : "permute_copy";
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
REGISTER_METAL_ACC(Squeeze, LAYER_UNSQUEEZE);

REGISTER_METAL_LAYOUT(LAYER_SQUEEZE, DATA_FORMAT_NC4HW4);
REGISTER_METAL_LAYOUT(LAYER_UNSQUEEZE, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS

