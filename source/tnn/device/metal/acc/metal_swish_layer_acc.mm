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
DECLARE_METAL_UNARY_ACC(Swish, LAYER_SWISH);

string MetalSwishLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "swish";
}

Status MetalSwishLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    return MetalUnaryLayerAcc::AllocateBufferParam(inputs, outputs);
}

Status MetalSwishLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalUnaryLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_UNARY_ACC(Swish, LAYER_SWISH);
REGISTER_METAL_LAYOUT(LAYER_SWISH, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
