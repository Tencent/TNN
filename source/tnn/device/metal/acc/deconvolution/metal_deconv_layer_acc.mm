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

#include "tnn/device/metal/acc/deconvolution/metal_deconv_layer_acc.h"
#include "tnn/device/metal/acc/deconvolution/metal_deconv_layer_common.h"
#include "tnn/device/metal/acc/deconvolution/metal_deconv_layer_depthwise.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"

namespace TNN_NS {

Status MetalDeconvLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (MetalDeconvLayerDepthwise::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        deconv_acc_impl_ = make_shared<MetalDeconvLayerDepthwise>();
    } else {
        deconv_acc_impl_ = make_shared<MetalDeconvLayerCommon>();
    }

    auto status = deconv_acc_impl_->Init(context, param, resource, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }

    return MetalLayerAcc::Init(context, param, resource, inputs, outputs);
}

MetalDeconvLayerAcc::~MetalDeconvLayerAcc() {}

Status MetalDeconvLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    do {
        if (MetalDeconvLayerDepthwise::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
            if (!deconv_acc_impl_ || !dynamic_cast<MetalDeconvLayerDepthwise *>(deconv_acc_impl_.get())) {
                auto deconv_acc = make_shared<MetalDeconvLayerDepthwise>();
                deconv_acc->Init(context_, param_, resource_, inputs, outputs);
                deconv_acc_impl_ = deconv_acc;
                break;
            }
        }
    } while (0);

    if (deconv_acc_impl_) {
        return deconv_acc_impl_->Reshape(inputs, outputs);
    } else {
        LOGE("Error: Deconv_acc_impl_ is nil\n");
        return Status(TNNERR_LAYER_ERR, "Deconv_acc_impl_ is nil");
    }
}

Status MetalDeconvLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (deconv_acc_impl_) {
        return deconv_acc_impl_->Forward(inputs, outputs);
    } else {
        return Status(TNNERR_LAYER_ERR, "Deconv_acc_impl_ is nil");
    }
}

REGISTER_METAL_ACC(Deconv, LAYER_DECONVOLUTION);
REGISTER_METAL_LAYOUT(LAYER_DECONVOLUTION, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
