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

#include "tnn/device/arm/acc/gradient/arm_sgd_layer_acc.h"

namespace TNN_NS {

Status ArmSGDLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);

    SGDParam *grad_param = dynamic_cast<SGDParam *>(param_);
    CHECK_PARAM_NULL(grad_param);
    learning_rate_ = grad_param->learning_rate;

    return TNN_OK;
}

ArmSGDLayerAcc::~ArmSGDLayerAcc() {}

Status ArmSGDLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    CHECK_PARAM_NULL(grad_info_);
    auto &trainables = grad_info_->trainable_resources;
    if (trainables.size() != inputs.size()) {
        LOGE("ArmSGDLayerAcc::DoForward, ERROR, grad and resource count not equal\n");
        return Status(TNNERR_NET_ERR, "grad and resource count not equal");
    }

    float *global_step_ptr_ = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
    if (!global_step_ptr_) {
        LOGE("ArmSGDLayerAcc::DoForward, ERROR, global_step is nil\n");
        return Status(TNNERR_NET_ERR, "global_step is nil");
    }

    *global_step_ptr_ = ++global_step_;

    LOGD("ArmSGDLayerAcc::DoForward, step: %d, lr: %f\n", int(*global_step_ptr_), learning_rate_);

    for (int i = 0; i < inputs.size(); ++i) {
        CHECK_PARAM_NULL(inputs[i]);
        CHECK_PARAM_NULL(trainables[i]);
        LOGD("Update: [%d] %s -> %d\n", i, inputs[i]->GetBlobDesc().description().c_str(),
             trainables[i]->GetDataCount());
        RETURN_ON_NEQ(ExecUpdate(inputs[i], trainables[i]), TNN_OK);
    }

    return TNN_OK;
}

Status ArmSGDLayerAcc::ExecUpdate(Blob *grad, RawBuffer *param) {
    if (param->GetDataType() != DATA_TYPE_FLOAT) {
        LOGE("ArmSGDLayerAcc::ExecUpdate ERROR, only support fp32 model now\n");
        return Status(TNNERR_LAYER_ERR, "SGD only support fp32 model now");
    }

    auto dims = grad->GetBlobDesc().dims;
    if (dims.size() != 2 || dims[0] != 1) {
        LOGE("ArmSGDLayerAcc::ExecUpdate ERROR, resource grad dims can not ignore data format\n");
        return Status(TNNERR_LAYER_ERR, "resource grad dims can not ignore data format");
    }
    if (DimsVectorUtils::Count(dims) != param->GetDataCount()) {
        LOGE("ArmSGDLayerAcc::ExecUpdate ERROR, grad and param data count not equal\n");
        return Status(TNNERR_LAYER_ERR, "grad and param data count not equal");
    }

    int count        = param->GetDataCount();
    float *grad_ptr  = reinterpret_cast<float *>(GetBlobHandlePtr(grad->GetHandle()));
    float *param_ptr = param->force_to<float *>();

    Float4 g0, g1;
    for (int n = 0; n < count - 3; n += 4) {
        g0 = Float4::load(grad_ptr + n);
        g1 = g0 * learning_rate_;
        Float4::save(param_ptr + n, Float4::load(param_ptr + n) - g1);
    }
    int remain = count % 4;
    grad_ptr += count << 2 >> 2;
    param_ptr += count << 2 >> 2;
    for (int n = 0; n < remain; ++n) {
        param_ptr[n] -= grad_ptr[n] * learning_rate_;
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(SGD, LAYER_SGD)
REGISTER_ARM_LAYOUT(LAYER_SGD, DATA_FORMAT_NCHW)
REGISTER_ARM_LAYOUT(LAYER_SGD, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
