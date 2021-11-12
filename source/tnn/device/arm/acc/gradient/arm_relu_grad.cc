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

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/train/gradient/layer_grad.h"

namespace TNN_NS {

DECLARE_ARM_LAYER_GRAD(Relu, LAYER_RELU);

Status ArmReluLayerGrad::OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                LayerResource *resource, LayerParam *param, Context *context,
                                LayerGradInfo *grad_info) {
    LOGD("ArmReluLayerGrad::OnGrad\n");

    return TNN_OK;
}

REGISTER_ARM_LAYER_GRAD(Relu, LAYER_RELU)
REGISTER_ARM_GRAD_LAYOUT(LAYER_RELU, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
