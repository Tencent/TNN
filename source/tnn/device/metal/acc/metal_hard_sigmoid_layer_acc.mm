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

namespace TNN_NS {

DECLARE_METAL_UNARY_ACC(HardSigmoid, LAYER_HARDSIGMOID);

string MetalHardSigmoidLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "hard_sigmoid";
}

Status MetalHardSigmoidLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                     const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<HardSigmoidLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: HardSigmoidLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "HardSigmoidLayerParam is nil");
    }

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalHardSigmoidParams metal_params;
        SetDefaultMetalParams(metal_params, dims_output, dims_output);

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

Status MetalHardSigmoidLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalUnaryLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_UNARY_ACC(HardSigmoid, LAYER_HARDSIGMOID);

} // namespace TNN_NS
