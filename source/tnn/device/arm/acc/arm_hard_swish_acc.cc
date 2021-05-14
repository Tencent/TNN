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

#include "tnn/device/arm/acc/arm_binary_layer_acc.h"

namespace TNN_NS {

DECLARE_ARM_BINARY_ACC(HardSwish);

Status ArmHardSwishLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = ArmBinaryLayerAcc::Init(context, param, resource, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }

    op_type_ = ArmBinaryOpType::kHARDSWISH;

    auto layer_param = dynamic_cast<HardSwishLayerParam *>(param);
    alpha_ = layer_param->alpha;
    beta_ = layer_param->beta;

    return TNN_OK;
}

ArmHardSwishLayerAcc::~ArmHardSwishLayerAcc() {}

REGISTER_ARM_ACC(HardSwish, LAYER_HARDSWISH)
REGISTER_ARM_LAYOUT(LAYER_HARDSWISH, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
