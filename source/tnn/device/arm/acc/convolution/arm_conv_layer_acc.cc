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

#include "tnn/device/arm/acc/convolution/arm_conv_layer_acc.h"

#include <memory>

#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_common.h"
#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_depthwise.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_1x1.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_3x3.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_c3.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_common.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_depthwise.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_depthwise_s1.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_group.h"
#include "tnn/interpreter/raw_buffer.h"

namespace TNN_NS {

static std::shared_ptr<LayerResource> CreateFp32ConvResource(ConvLayerResource *conv_f16) {
    ConvLayerResource *conv_f32 = new ConvLayerResource();

    conv_f32->filter_handle = ConvertHalfHandle(conv_f16->filter_handle);
    conv_f32->scale_handle  = ConvertHalfHandle(conv_f16->scale_handle);
    conv_f32->bias_handle   = ConvertHalfHandle(conv_f16->bias_handle);

    return std::shared_ptr<LayerResource>(conv_f32);
}

Status ArmConvLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                             const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret;
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource);
    CHECK_PARAM_NULL(conv_res);

    if (conv_res->filter_handle.GetDataType() == DATA_TYPE_HALF) {
        conv_acc_f32_resource_ = CreateFp32ConvResource(conv_res);
        ret                    = ArmLayerAcc::Init(context, param, conv_acc_f32_resource_.get(), inputs, outputs);
    } else {
        ret = ArmLayerAcc::Init(context, param, resource, inputs, outputs);
    }
    if (ret != TNN_OK)
        return ret;

    auto data_type = inputs[0]->GetBlobDesc().data_type;
    if (conv_param->group != 1 && conv_param->group != inputs[0]->GetBlobDesc().dims[1]) {
        conv_acc_impl_ = std::make_shared<ArmConvLayerGroup>();
    } else {
        if (data_type == DATA_TYPE_INT8) {
            GetImpInt8(inputs, outputs);
        } else {
            GetImpFP(inputs, outputs);
        }
    }

    if (!conv_acc_impl_) {
        return Status(TNNERR_NET_ERR, "Could not create conv impl_");
    }
    return conv_acc_impl_->Init(context_, param_, resource_, inputs, outputs);
}

ArmConvLayerAcc::~ArmConvLayerAcc() {}

/*
get different impl based on conv params
ArmConvInt8LayerCommon always as the last solution
*/
void ArmConvLayerAcc::GetImpInt8(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (ArmConvInt8LayerDepthwise::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
        if (!dynamic_cast<ArmConvInt8LayerDepthwise *>(conv_acc_impl_.get())) {
            conv_acc_impl_ = std::make_shared<ArmConvInt8LayerDepthwise>();
        }
    } else if (ArmConvInt8LayerCommon::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
        if (!dynamic_cast<ArmConvInt8LayerCommon *>(conv_acc_impl_.get())) {
            conv_acc_impl_ = std::make_shared<ArmConvInt8LayerCommon>();
        }
    }
}

/*
get different impl based on conv params
ArmConvLayerCommon always as the last solution
bfp16 impl included in fp impl
*/
void ArmConvLayerAcc::GetImpFP(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (ArmConvLayerC3::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
        if (!dynamic_cast<ArmConvLayerC3 *>(conv_acc_impl_.get())) {
            conv_acc_impl_ = std::make_shared<ArmConvLayerC3>();
        }
    } else if (ArmConvLayer3x3::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
        if (!dynamic_cast<ArmConvLayer3x3 *>(conv_acc_impl_.get())) {
            conv_acc_impl_ = std::make_shared<ArmConvLayer3x3>();
        }
    } else if (ArmConvLayer1x1::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
        if (!dynamic_cast<ArmConvLayer1x1 *>(conv_acc_impl_.get())) {
            conv_acc_impl_ = std::make_shared<ArmConvLayer1x1>();
        }
    } else if (ArmConvLayerDepthwise::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
        if (ArmConvLayerDepthwiseS1::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
            if (!dynamic_cast<ArmConvLayerDepthwiseS1 *>(conv_acc_impl_.get())) {
                conv_acc_impl_ = std::make_shared<ArmConvLayerDepthwiseS1>();
            }
        } else if (!dynamic_cast<ArmConvLayerDepthwise *>(conv_acc_impl_.get())) {
            conv_acc_impl_ = std::make_shared<ArmConvLayerDepthwise>();
        }
    }
    if (!conv_acc_impl_) {
        conv_acc_impl_ = std::make_shared<ArmConvLayerCommon>();
    }
}

Status ArmConvLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return conv_acc_impl_->Reshape(inputs, outputs);
}

Status ArmConvLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (conv_acc_impl_) {
        return conv_acc_impl_->DoForward(inputs, outputs);
    } else {
        return Status(TNNERR_CONTEXT_ERR, "conv_acc_impl_ is nil");
    }
}

REGISTER_ARM_ACC(Conv, LAYER_CONVOLUTION)

}  // namespace TNN_NS
