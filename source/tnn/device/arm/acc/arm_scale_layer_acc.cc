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

#include "tnn/device/arm/acc/arm_batch_norm_layer_acc.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {
// @brief conv layer cpu acc
class ArmScaleLayerAcc : public ArmBatchNormLayerAcc {
public:
    virtual ~ArmScaleLayerAcc(){};
};

REGISTER_ARM_ACC(Scale, LAYER_SCALE)
REGISTER_ARM_PRECISION_FP16(LAYER_SCALE)
REGISTER_ARM_LAYOUT(LAYER_SCALE, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
