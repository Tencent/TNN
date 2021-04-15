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
#include "tnn/device/metal/acc/metal_unary_layer_acc.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
DECLARE_METAL_UNARY_ACC(Clip, LAYER_CLIP);

string MetalClipLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "clip";
}

Status MetalClipLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<ClipLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: ClipLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "ClipLayerParam is nil");
    }

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalClipParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        FixDefaultMetalParams(metal_params, dims_input, dims_output);

        metal_params.min = layer_param->min;
        metal_params.max = layer_param->max;

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalClipParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalClipLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalUnaryLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_UNARY_ACC(Clip, LAYER_CLIP);
REGISTER_METAL_LAYOUT(LAYER_CLIP, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
