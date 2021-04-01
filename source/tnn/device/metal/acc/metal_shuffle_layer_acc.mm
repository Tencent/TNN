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

Status IsMetalShuffleLayerAccSupported(LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                                       const std::vector<Blob *> &outputs) {

    auto layer_param = dynamic_cast<ShuffleLayerParam *>(param);
    if (!layer_param || layer_param->group <= 0) {
        LOGE("ShuffleLayerParam is nil\n");
        return Status(TNNERR_LAYER_ERR, "ShuffleLayerParam is nil");
    }
    auto dims_input = inputs[0]->GetBlobDesc().dims;
    if (dims_input[1] % layer_param->group != 0) {
        LOGE("ShuffleLayerParam group is invalid\n");
        return Status(TNNERR_LAYER_ERR, "ShuffleLayerParam group is invalid");
    }

    return TNN_OK;
}

DECLARE_METAL_ACC(Shuffle, LAYER_SHUFFLE_CHANNEL);

Status MetalShuffleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalShuffleLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    Status status = IsMetalShuffleLayerAccSupported(param_, resource_, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    auto layer_param = dynamic_cast<ShuffleLayerParam *>(param_);

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_output = outputs[0]->GetBlobDesc().dims;
    {
        auto dims_input = inputs[0]->GetBlobDesc().dims;
        MetalShuffleParams metal_params;

        metal_params.input_size    = DimsFunctionUtils::GetDimProduct(dims_input, 2);
        metal_params.input_channel = dims_input[1];
        metal_params.input_slice   = UP_DIV(dims_input[1], 4);

        metal_params.output_size  = DimsFunctionUtils::GetDimProduct(dims_output, 2);
        metal_params.output_slice = UP_DIV(dims_output[1], 4);

        metal_params.group             = layer_param->group;
        metal_params.channel_per_group = dims_input[1] / metal_params.group;

        metal_params.batch = dims_output[0];

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalShuffleParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    return TNN_OK;
}

std::string MetalShuffleLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "channel_shuffle";
}

Status MetalShuffleLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    const auto& output_dims = outputs[0]->GetBlobDesc().dims;
    auto hw = DimsVectorUtils::Count(output_dims, 2);
    auto slice = UP_DIV(output_dims[1], 4);
    size = MTLSizeMake(hw, slice, output_dims[0]);

    return TNN_OK;
    //return MetalLayerAcc::ComputeThreadSize(inputs, outputs, size);
}

Status MetalShuffleLayerAcc::SetKernelEncoderParam(
                                               id<MTLComputeCommandEncoder> encoder,
                                               const std::vector<Blob *> &inputs,
                                               const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

Status MetalShuffleLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = IsMetalShuffleLayerAccSupported(param_, resource_, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(Shuffle, LAYER_SHUFFLE_CHANNEL);
REGISTER_METAL_LAYOUT(LAYER_SHUFFLE_CHANNEL, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
