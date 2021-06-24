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

#include "tnn/device/arm/acc/convolution/arm_conv_layer_acc_factory.h"

namespace TNN_NS {

/*
get different impl based on conv params
ArmConvInt8LayerCommon always as the last solution
*/
void ArmConvLayerAccFactory::CreateImpInt8(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                           LayerParam *param, std::shared_ptr<ArmLayerAcc> &conv_acc_impl) {
    if (0) {
    }
#ifdef TNN_ARM82_USE_NEON
    else if (ArmConvInt8SdotLayerDepthwise3x3::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<ArmConvInt8SdotLayerDepthwise3x3 *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvInt8SdotLayerDepthwise3x3>();
        }
    }
#endif
    else if (ArmConvInt8LayerDepthwise::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<ArmConvInt8LayerDepthwise *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvInt8LayerDepthwise>();
        }
    }
#ifdef TNN_ARM82_USE_NEON
    else if (ArmConvInt8SdotLayerCommon::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<ArmConvInt8SdotLayerCommon *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvInt8SdotLayerCommon>();
        }
    }
#endif
    else if (ArmConvInt8Layer1x1::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<ArmConvInt8Layer1x1 *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvInt8Layer1x1>();
        }
    } else if (ArmConvInt8LayerCommon::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<ArmConvInt8LayerCommon *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvInt8LayerCommon>();
        }
    }
}

/*
get different impl based on conv params
ArmConvLayerCommon always as the last solution
bfp16 impl included in fp impl
*/
void ArmConvLayerAccFactory::CreateImpFP(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                         LayerParam *param, std::shared_ptr<ArmLayerAcc> &conv_acc_impl) {
    if (ArmConvLayerC3::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<ArmConvLayerC3 *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvLayerC3>();
        }
    } else if (ArmConvLayer3x3::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<ArmConvLayer3x3 *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvLayer3x3>();
        }
    } else if (ArmConvLayer1x1::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<ArmConvLayer1x1 *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvLayer1x1>();
        }
    } else if (ArmConvLayerDepthwise::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (ArmConvLayerDepthwiseS1::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
            if (!dynamic_cast<ArmConvLayerDepthwiseS1 *>(conv_acc_impl.get())) {
                conv_acc_impl = std::make_shared<ArmConvLayerDepthwiseS1>();
            }
        } else if (!dynamic_cast<ArmConvLayerDepthwise *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvLayerDepthwise>();
        }
    }
    if (!conv_acc_impl) {
        conv_acc_impl = std::make_shared<ArmConvLayerCommon>();
    }
}

#if TNN_ARM82
void ArmConvLayerAccFactory::CreateImpHalf(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                         LayerParam *param, std::shared_ptr<ArmLayerAcc> &conv_acc_impl) {
    if (ArmConvFp16LayerDepthwise::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (ArmConvFp16LayerDepthwiseS1::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
            if (!dynamic_cast<ArmConvFp16LayerDepthwiseS1 *>(conv_acc_impl.get())) {
                conv_acc_impl = std::make_shared<ArmConvFp16LayerDepthwiseS1>();
            }
        } else {
            if (!dynamic_cast<ArmConvFp16LayerDepthwise *>(conv_acc_impl.get())) {
                conv_acc_impl = std::make_shared<ArmConvFp16LayerDepthwise>();
            }
        }
    } else if (ArmConvFp16LayerC3::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<ArmConvFp16LayerC3 *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvFp16LayerC3>();
        }
    } else if (ArmConvFp16Layer3x3::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<ArmConvFp16Layer3x3 *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvFp16Layer3x3>();
        }
    } else if (ArmConvFp16LayerCommon::isPrefered(dynamic_cast<ConvLayerParam *>(param), inputs, outputs)) {
        if (!dynamic_cast<ArmConvFp16LayerCommon *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvFp16LayerCommon>();
        }
    }
}
#endif

}  // namespace TNN_NS
