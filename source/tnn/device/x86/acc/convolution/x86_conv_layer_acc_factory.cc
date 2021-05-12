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

#include "tnn/device/x86/acc/convolution/x86_conv_layer_acc_factory.h"

#include "tnn/device/x86/acc/convolution/x86_conv_layer_depthwise.h"
#include "tnn/device/x86/acc/convolution/x86_conv_layer_1x1.h"
#include "tnn/device/x86/acc/convolution/x86_conv_layer_3x3.h"

namespace TNN_NS {

/*
get different impl based on conv params
ArmConvLayerCommon always as the last solution
bfp16 impl included in fp impl
*/
void X86ConvLayerAccFactory::CreateImpFP(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                         LayerParam *param, std::shared_ptr<X86LayerAcc> &conv_acc_impl) {
    if (X86ConvLayerDepthwise::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<X86ConvLayerDepthwise *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<X86ConvLayerDepthwise>();
        }
    } else if (X86ConvLayer1x1::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<X86ConvLayer1x1*>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<X86ConvLayer1x1>();
        }
    } else if (X86ConvLayer3x3::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<X86ConvLayer3x3*>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<X86ConvLayer3x3>();
        }
    } else if (!conv_acc_impl) {
        conv_acc_impl = std::make_shared<X86ConvLayerCommon>();
    }
}

}  // namespace TNN_NS
