// Tencent is pleased to support the open source community by making TNN
// available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "tnn/device/arm/acc/deconvolution/arm_deconv_layer_acc.h"

#include <memory>

#include "tnn/device/arm/acc/deconvolution/arm_deconv_layer_stride.h"
#include "tnn/device/arm/acc/deconvolution/arm_deconv_layer_common.h"
#include "tnn/device/arm/acc/deconvolution/arm_deconv_layer_depthwise.h"

namespace TNN_NS {

static std::shared_ptr<LayerResource> CreateFp32DeconvResource(ConvLayerResource *deconv_f16) {
    ConvLayerResource *deconv_f32 = new ConvLayerResource();

    deconv_f32->filter_handle = ConvertHalfHandle(deconv_f16->filter_handle);
    deconv_f32->scale_handle  = ConvertHalfHandle(deconv_f16->scale_handle);
    deconv_f32->bias_handle   = ConvertHalfHandle(deconv_f16->bias_handle);

    return std::shared_ptr<LayerResource>(deconv_f32);
}

Status ArmDeconvLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret;
    ConvLayerParam *deconv_param = dynamic_cast<ConvLayerParam *>(param);
    CHECK_PARAM_NULL(deconv_param);

    ConvLayerResource *deconv_res = dynamic_cast<ConvLayerResource *>(resource);
    CHECK_PARAM_NULL(deconv_res);

    if (deconv_res->filter_handle.GetDataType() == DATA_TYPE_HALF) {
        deconv_acc_f32_resource_ = CreateFp32DeconvResource(deconv_res);
        ret                      = ArmLayerAcc::Init(context, param, deconv_acc_f32_resource_.get(), inputs, outputs);
    } else {
        ret = ArmLayerAcc::Init(context, param, resource, inputs, outputs);
    }

    if (ret != TNN_OK) {
        return ret;
    }

    auto data_type = inputs[0]->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_BFP16) {
        GetImpFP(inputs, outputs);
    } else {
        return Status(TNNERR_NET_ERR, "int8 deconv impl is not supported");
    }

    if (!deconv_acc_impl_) {
        return Status(TNNERR_NET_ERR, "Could not create conv impl_");
    }

    return deconv_acc_impl_->Init(context_, param_, resource_, inputs, outputs);
}

ArmDeconvLayerAcc::~ArmDeconvLayerAcc() {}

void ArmDeconvLayerAcc::GetImpFP(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (ArmDeconvLayerDepthwise::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
        if (!deconv_acc_impl_ || !dynamic_cast<ArmDeconvLayerDepthwise *>(deconv_acc_impl_.get())) {
            auto deconv_acc  = std::make_shared<ArmDeconvLayerDepthwise>();
            deconv_acc_impl_ = deconv_acc;
        }
    } else if (ArmDeconvLayerStride::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
        if (!dynamic_cast<ArmDeconvLayerStride*>(deconv_acc_impl_.get())) {
            deconv_acc_impl_ = std::make_shared<ArmDeconvLayerStride>();
        }
    }
    if (!deconv_acc_impl_) {
        deconv_acc_impl_ = std::make_shared<ArmDeconvLayerCommon>();
    }
}

Status ArmDeconvLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return deconv_acc_impl_->Reshape(inputs, outputs);
}

Status ArmDeconvLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (deconv_acc_impl_) {
        return deconv_acc_impl_->DoForward(inputs, outputs);
    } else {
        return Status(TNNERR_CONTEXT_ERR, "deconv_acc_impl_ is nil");
    }
}

REGISTER_ARM_ACC(Deconv, LAYER_DECONVOLUTION)

}  // namespace TNN_NS
