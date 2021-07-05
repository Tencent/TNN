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

#include "tnn/device/metal/acc/convolution/metal_conv_layer_acc.h"
#include "tnn/device/metal/acc/convolution/metal_conv_layer_1x1.h"
#include "tnn/device/metal/acc/convolution/metal_conv_layer_common.h"
#include "tnn/device/metal/acc/convolution/metal_conv_layer_depthwise.h"
#include "tnn/device/metal/acc/convolution/metal_conv_layer_winograd.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"

namespace TNN_NS {

Status MetalConvLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (MetalConvLayerDepthwise::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        conv_acc_impl_ = make_shared<MetalConvLayerDepthwise>();
    } else if (MetalConvLayer1x1::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        conv_acc_impl_ = make_shared<MetalConvLayer1x1>();
    } else if (MetalConvLayerWinograd::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        conv_acc_impl_ = make_shared<MetalConvLayerWinograd>();
    } else {
        conv_acc_impl_ = make_shared<MetalConvLayerCommon>();
    }

    auto status = conv_acc_impl_->Init(context, param, resource, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }

    return MetalLayerAcc::Init(context, param, resource, inputs, outputs);
}

MetalConvLayerAcc::~MetalConvLayerAcc() {
    conv_acc_impl_ = nullptr;
}

Status MetalConvLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    do {
        if (MetalConvLayerDepthwise::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
            if (!conv_acc_impl_ || !dynamic_cast<MetalConvLayerDepthwise *>(conv_acc_impl_.get())) {
                auto conv_acc = make_shared<MetalConvLayerDepthwise>();
                conv_acc->Init(context_, param_, resource_, inputs, outputs);
                conv_acc_impl_ = conv_acc;
            }
            break;
        }

        if (MetalConvLayer1x1::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
            if (!conv_acc_impl_ || !dynamic_cast<MetalConvLayer1x1 *>(conv_acc_impl_.get())) {
                auto conv_acc = make_shared<MetalConvLayer1x1>();
                conv_acc->Init(context_, param_, resource_, inputs, outputs);
                conv_acc_impl_ = conv_acc;
            }
            break;
        }

        if (MetalConvLayerWinograd::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
            if (!conv_acc_impl_ || !dynamic_cast<MetalConvLayerWinograd *>(conv_acc_impl_.get())) {
                auto conv_acc = make_shared<MetalConvLayerWinograd>();
                conv_acc->Init(context_, param_, resource_, inputs, outputs);
                conv_acc_impl_ = conv_acc;
            }
            break;
        }
    } while (0);

    if (conv_acc_impl_) {
        return conv_acc_impl_->Reshape(inputs, outputs);
    } else {
        LOGE("Error: conv_acc_impl_ is nil\n");
        return Status(TNNERR_LAYER_ERR, "conv_acc_impl_ is nil");
    }
}

Status MetalConvLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (conv_acc_impl_) {
        return conv_acc_impl_->Forward(inputs, outputs);
    } else {
        return Status(TNNERR_LAYER_ERR, "conv_acc_impl_ is nil");
    }
}

REGISTER_METAL_ACC(Conv, LAYER_CONVOLUTION);
REGISTER_METAL_LAYOUT(LAYER_CONVOLUTION, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
