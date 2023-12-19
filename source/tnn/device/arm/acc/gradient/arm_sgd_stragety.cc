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

DECLARE_ARM_SOLVER_STRATEGY(SGD, SOLVER_TYPE_SGD)

Status ArmSGDSolverStrategy::ExecUpdate(Blob *grad, RawBuffer *resource, SolverParam *param, Context *context) {
    const BlobDesc& grad_desc = grad->GetBlobDesc();
    
    if (resource->GetDataType() != DATA_TYPE_FLOAT || grad_desc.data_type != DATA_TYPE_FLOAT) {
        LOGE("ArmSGDSolverStrategy::ExecUpdate ERROR, only support fp32 model now\n");
        return Status(TNNERR_TRAIN_ERROR, "solver only support fp32 model now");
    }

    float *resource_ptr = resource->force_to<float *>();
    float *grad_ptr = grad->GetHandle().force_to<float *>();;

    if (grad_desc.data_format != DATA_FORMAT_NCHW && grad_desc.data_format != DATA_FORMAT_NC4HW4) {
        return Status(TNNERR_TRAIN_ERROR, "grad only support nchw and nc4hw4");
    }
    
    RawBuffer grad_nc4hw4;
    if (grad_desc.data_format == DATA_FORMAT_NC4HW4 && !FloatBlobCanIgnorePack(grad_desc.dims)) {
        grad_nc4hw4 = RawBuffer(DimsVectorUtils::Count(grad_desc.dims) * sizeof(float));
        grad_ptr = grad_nc4hw4.force_to<float *>();
        UnpackFloatBlob(grad_ptr, grad->GetHandle().force_to<float *>(), grad_desc.dims);
    }

    auto learning_rate  = param->learning_rate;
    int count = resource->GetDataCount();
    for (int n = 0; n < count - 3; n += 4) {
        Float4 g = Float4::load(grad_ptr + n) * learning_rate;
        Float4::save(resource_ptr + n, Float4::load(resource_ptr + n) - g);
    }
    int remain = count % 4;
    grad_ptr += count >> 2 << 2;
    resource_ptr += count >> 2 << 2;
    for (int n = 0; n < remain; ++n) {
        resource_ptr[n] -= grad_ptr[n] * learning_rate;
    }

    return TNN_OK;
}

REGISTER_ARM_SOLVER_STRATEGY(SGD, SOLVER_TYPE_SGD)

}  // namespace TNN_NS
