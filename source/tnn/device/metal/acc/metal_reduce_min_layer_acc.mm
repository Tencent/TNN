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

#include "tnn/device/metal/acc/metal_reduce_layer_acc.h"

namespace TNN_NS {

DECLARE_METAL_REDUCE_ACC(ReduceMin, LAYER_REDUCE_MIN);

std::string MetalReduceMinLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (axis_ == 0) {
        return "reduce_min_axis_0_common";
    } else if (axis_ == 1) {
        return "reduce_min_axis_1_common";
    } else if (axis_ == 2) {
        return "reduce_min_axis_2_common";
    } else {
        return "reduce_min_axis_3_common";
    }
}

Status MetalReduceMinLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalReduceLayerAcc::Reshape(inputs, outputs);
}

Status MetalReduceMinLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalReduceLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_REDUCE_ACC(ReduceMin, LAYER_REDUCE_MIN);

} // namespace TNN_NS