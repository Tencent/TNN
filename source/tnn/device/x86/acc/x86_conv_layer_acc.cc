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

#include "x86_conv_layer_acc.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/device/x86/acc/convolution/x86_conv_layer_acc_factory.h"

namespace TNN_NS {

Status X86ConvLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                             const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto conv_param    = dynamic_cast<ConvLayerParam *>(param);
    CHECK_PARAM_NULL(conv_param);
    auto conv_resource = dynamic_cast<ConvLayerResource *>(resource);
    CHECK_PARAM_NULL(conv_resource);

    Status ret = X86LayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }

    X86ConvLayerAccFactory::CreateImpFP(inputs, outputs, param_, conv_acc_impl_);

    if (!conv_acc_impl_) {
        return Status(TNNERR_NET_ERR, "Could not create conv impl_");
    }
    return conv_acc_impl_->Init(context_, param_, resource_, inputs, outputs);

    return TNN_OK;
}

Status X86ConvLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status X86ConvLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (conv_acc_impl_) {
        return conv_acc_impl_->DoForward(inputs, outputs);
    } else {
        return Status(TNNERR_CONTEXT_ERR, "conv_acc_impl_ is nil");
    }
}

REGISTER_X86_ACC(Conv, LAYER_CONVOLUTION);

}   // namespace TNN_NS