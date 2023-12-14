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

#include "tnn/device/arm/acc/gradient/arm_gradient_layer_acc.h"

namespace TNN_NS {

Status ArmGradientLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    grad_param_ = dynamic_cast<GradientParam *>(param);
    CHECK_PARAM_NULL(grad_param_);

    impl_ = GradOp::CreateGradOp(DEVICE_ARM, grad_param_->forward_layer_type);
    if (!impl_) {
        LOGE("ArmGradientLayerAcc::Init ERROR, layer grad not implemented: %d\n", grad_param_->forward_layer_type);
        return Status(TNNERR_TRAIN_ERROR, "layer grad not implemented");
    }

    return ArmLayerAcc::Init(context, param, resource, inputs, outputs);
}

ArmGradientLayerAcc::~ArmGradientLayerAcc() {}

Status ArmGradientLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    CHECK_PARAM_NULL(impl_);
    if (!runtime_training_info_) {
        LOGE("ArmGradientLayerAcc::DoForward ERROR, runtime training info is nil\n");
        return Status(TNNERR_TRAIN_ERROR, "runtime training info is nil");
    }

    return impl_->OnGrad(inputs, outputs, resource_, grad_param_, context_, runtime_training_info_->grad_info);
}

REGISTER_ARM_ACC(Gradient, LAYER_GRADIENT)

}  // namespace TNN_NS
