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

#include "x86_conv1d_layer_acc.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/device/x86/acc/convolution/x86_conv_int8_layer_common.h"
#include "tnn/device/x86/acc/convolution/x86_conv_layer_common.h"
#include "tnn/interpreter/layer_resource_generator.h"

namespace TNN_NS {

Status X86Conv1DLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                             const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto conv1d_param  = dynamic_cast<ConvLayerParam *>(param);
    CHECK_PARAM_NULL(conv1d_param);
    auto conv_resource = dynamic_cast<ConvLayerResource *>(resource);
    CHECK_PARAM_NULL(conv_resource);

    ConvLayerParam *conv2d_param = new ConvLayerParam(*conv1d_param);
    // Fill up the parameters of conv1d to conv2d.
    // do not modify param, otherwise it will cause reshape and get wrong output dims
    conv2d_param->kernels.insert(conv2d_param->kernels.begin(), 1);
    conv2d_param->strides.insert(conv2d_param->strides.begin(), 1);
    conv2d_param->dialations.insert(conv2d_param->dialations.begin(), 1);
    conv2d_param->pads.insert(conv2d_param->pads.begin(), 2, 0);

    Status ret;
    if (conv_resource->filter_handle.GetDataType() == DATA_TYPE_HALF) {
        LayerResource *fp32_res = nullptr;
        RETURN_ON_NEQ(ConvertHalfResource(LAYER_CONVOLUTION_1D, conv_resource, &fp32_res), TNN_OK);
        conv_acc_f32_resource_ = std::shared_ptr<LayerResource>(fp32_res);
        ret = X86LayerAcc::Init(context, conv2d_param, conv_acc_f32_resource_.get(), inputs, outputs);
    } else {
        ret = X86LayerAcc::Init(context, conv2d_param, resource, inputs, outputs);
    }

    if (ret != TNN_OK) {
        return ret;
    }

    auto data_type = inputs[0]->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_FLOAT) {
        conv_acc_impl_ = std::make_shared<X86ConvLayerCommon>();
    } else {
        return Status(TNNERR_LAYER_ERR, "Conv1D only support float datatype");
    }

    if (!conv_acc_impl_) {
        return Status(TNNERR_NET_ERR, "Could not create conv impl_");
    }
    ret = conv_acc_impl_->Init(context_, param_, resource_, inputs, outputs);

    // converted weights are assumed to be packed, and can be freed now
    if (conv_acc_f32_resource_) {
        conv_acc_f32_resource_.reset();
        resource_ = nullptr;
    }

    return ret;
}

Status X86Conv1DLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (conv_acc_impl_) {
        return conv_acc_impl_->DoForward(inputs, outputs);
    } else {
        return Status(TNNERR_CONTEXT_ERR, "conv_acc_impl_ is nil");
    }
}

REGISTER_X86_ACC(Conv1D, LAYER_CONVOLUTION_1D);

}   // namespace TNN_NS