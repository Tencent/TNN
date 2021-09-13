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

#include "tnn/device/arm/acc/convolution/arm_conv1d_layer_acc.h"

#include <memory>

#include "tnn/device/arm/acc/convolution/arm_conv_layer_common.h"
#if TNN_ARM82
#include "tnn/device/arm/acc/convolution/arm_conv_fp16_layer_common.h"
#endif
#include "tnn/interpreter/layer_resource_generator.h"
#include "tnn/interpreter/raw_buffer.h"

namespace TNN_NS {

Status ArmConv1DLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                             const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret;
    ConvLayerParam *conv1d_param = dynamic_cast<ConvLayerParam *>(param);
    CHECK_PARAM_NULL(conv1d_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource);
    CHECK_PARAM_NULL(conv_res);

    ConvLayerParam *conv2d_param = new ConvLayerParam(*conv1d_param);
    // Fill up the parameters of conv1d to conv2d.
    // do not modify param, otherwise it will cause reshape and get wrong output dims
    conv2d_param->kernels.insert(conv2d_param->kernels.begin(), 1);
    conv2d_param->strides.insert(conv2d_param->strides.begin(), 1);
    conv2d_param->dialations.insert(conv2d_param->dialations.begin(), 1);
    conv2d_param->pads.insert(conv2d_param->pads.begin(), 2, 0);

    if (conv_res->filter_handle.GetDataType() == DATA_TYPE_HALF) {
        LayerResource *fp32_res = nullptr;
        RETURN_ON_NEQ(ConvertHalfResource(LAYER_CONVOLUTION_1D, conv_res, &fp32_res), TNN_OK);
        conv_acc_f32_resource_ = std::shared_ptr<LayerResource>(fp32_res);
        ret                    = ArmLayerAcc::Init(context, conv2d_param, conv_acc_f32_resource_.get(), inputs, outputs);
    } else {
        ret = ArmLayerAcc::Init(context, conv2d_param, resource, inputs, outputs);
    }
    if (ret != TNN_OK) {
        return ret;
    }
    auto data_type = inputs[0]->GetBlobDesc().data_type;

    if (data_type == DATA_TYPE_FLOAT) {
        conv_acc_impl_ = std::make_shared<ArmConvLayerCommon>();
    }
#if TNN_ARM82
    else if (data_type == DATA_TYPE_HALF) {
        conv_acc_impl_ = std::make_shared<ArmConvFp16LayerCommon>();
    }
#endif
    else {
        return Status(TNNERR_LAYER_ERR, "Conv1D only support fp32 / fp16 datatype");
    }

    if (!conv_acc_impl_) {
        return Status(TNNERR_NET_ERR, "Could not create conv impl_");
    }
    return conv_acc_impl_->Init(context_, param_, resource_, inputs, outputs);
}

ArmConv1DLayerAcc::~ArmConv1DLayerAcc() {}

Status ArmConv1DLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // converted weights are assumed to be packed, and can be freed now
    if (conv_acc_f32_resource_) {
        conv_acc_f32_resource_.reset();
    }

    if (conv_acc_impl_) {
        return conv_acc_impl_->DoForward(inputs, outputs);
    } else {
        return Status(TNNERR_CONTEXT_ERR, "conv_acc_impl_ is nil");
    }
}

REGISTER_ARM_ACC(Conv1D, LAYER_CONVOLUTION_1D)
REGISTER_ARM_PRECISION_FP16(LAYER_CONVOLUTION_1D)
REGISTER_ARM_LAYOUT(LAYER_CONVOLUTION_1D, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
