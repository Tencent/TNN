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

namespace TNN_NS {
DECLARE_METAL_MULTIDIR_BROADCAST_ACC(Add, LAYER_ADD);

std::string MetalAddLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    std::string kernel_name = "";
    auto layer_param        = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return kernel_name;
    }
    if (layer_param->input0_broadcast_type > 0 || layer_param->input1_broadcast_type > 0) {
        kernel_name = "add_broadcast";
    } else {
        kernel_name = "add_normal";
    }
    return kernel_name;
}

Status MetalAddLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalMultidirBroadcastLayerAcc::Reshape(inputs, outputs);
}

Status MetalAddLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalMultidirBroadcastLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_MULTIDIR_BROADCAST_ACC(Add, LAYER_ADD);

} // namespace TNN_NS
