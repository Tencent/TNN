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

DECLARE_METAL_ACC(Permute, LAYER_PERMUTE);

Status MetalPermuteLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalPermuteLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    return  MetalLayerAcc::AllocateBufferParam(inputs, outputs);
}

Status MetalPermuteLayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

std::string MetalPermuteLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PermuteLayerParam *>(param_);
    if (layer_param->orders[0] == 0 && layer_param->orders[1] == 2 &&
        layer_param->orders[2] == 3 && layer_param->orders[3] == 1) {
        return "permute_to_nhwc";
    } else if (layer_param->orders[0] == 0 && layer_param->orders[1] == 2 &&
               layer_param->orders[2] == 1 &&  layer_param->orders[3] == 3) {
        return "permute_to_nhcw";
    } else if (layer_param->orders[0] == 0 && layer_param->orders[1] == 3 &&
               layer_param->orders[2] == 1 &&  layer_param->orders[3] == 2) {
        return "permute_to_nwch";
    } else if (layer_param->orders[0] == 1 && layer_param->orders[1] == 2 &&
               layer_param->orders[2] == 3 &&  layer_param->orders[3] == 0) {
        return "permute_to_chwn";
    } else if (layer_param->orders[0] == 0 && layer_param->orders[1] == 1 &&
               layer_param->orders[2] == 2 &&  layer_param->orders[3] == 3) {
        return "permute_copy";
    }
    return "";
}

Status MetalPermuteLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, false);
    return TNN_OK;
}

Status MetalPermuteLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PermuteLayerParam *>(param_);
    if (!layer_param || layer_param->orders.size() < 4) {
        return Status(TNNERR_PARAM_ERR, "PermuteLayerParam is nil");
    }
    
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(Permute, LAYER_PERMUTE);

} // namespace TNN_NS
