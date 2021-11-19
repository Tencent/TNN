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

#include "tnn/device/arm/acc/gradient/arm_solver_layer_acc.h"

namespace TNN_NS {

Status ArmSolverLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    solver_param_ = dynamic_cast<SolverParam *>(param);
    CHECK_PARAM_NULL(solver_param_);

    impl_ = SolverStrategy::GetSolverStrategy(DEVICE_ARM, solver_param_->type);
    if (!impl_) {
        LOGE("ArmSolverLayerAcc::Init ERROR, solver strategy not implemented: %d\n", solver_param_->type);
        return Status(TNNERR_TRAIN_ERROR, "solver strategy not implemented");
    }

    return ArmLayerAcc::Init(context, param, resource, inputs, outputs);
}

ArmSolverLayerAcc::~ArmSolverLayerAcc() {}

Status ArmSolverLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    CHECK_PARAM_NULL(impl_);
    if (!runtime_training_info_) {
        LOGE("ArmSolverLayerAcc::DoForward ERROR, runtime training info is nil\n");
        return Status(TNNERR_TRAIN_ERROR, "runtime training info is nil");
    }

    float *global_step_init_ptr_ = reinterpret_cast<float *>(GetBlobHandlePtr(inputs.back()->GetHandle()));
    if (!global_step_init_ptr_) {
        LOGE("ArmSolverLayerAcc::DoForward, ERROR, global_step_init is nil\n");
        return Status(TNNERR_NET_ERR, "global_step_init is nil");
    }
    float *global_step_ptr_ = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
    if (!global_step_ptr_) {
        LOGE("ArmSolverLayerAcc::DoForward, ERROR, global_step is nil\n");
        return Status(TNNERR_NET_ERR, "global_step is nil");
    }
    *global_step_init_ptr_ = *global_step_init_ptr_ + 1;
    *global_step_ptr_      = *global_step_init_ptr_;
    LOGD("ArmSolverLayerAcc::DoForward, step: %d, lr: %f\n", int(*global_step_ptr_), solver_param_->learning_rate);

    return impl_->OnSolve(inputs, outputs, solver_param_, context_, runtime_training_info_->solver_info);
}

REGISTER_ARM_ACC(Solver, LAYER_SOLVER)
// REGISTER_ARM_LAYOUT(LAYER_SOLVER, DATA_FORMAT_NCHW)
// REGISTER_ARM_LAYOUT(LAYER_SOLVER, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
