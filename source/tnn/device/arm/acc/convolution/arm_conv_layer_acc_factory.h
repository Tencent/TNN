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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_CONV_LAYER_ACC_FACTORY_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_CONV_LAYER_ACC_FACTORY_H_

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_1x1.h"
#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_common.h"
#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_depthwise.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_1x1.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_3x3.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_c3.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_common.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_depthwise.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_depthwise_s1.h"

namespace TNN_NS {

class ArmConvLayerAccFactory {
public:
    static void CreateImpInt8(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs, LayerParam *param,
                              std::shared_ptr<ArmLayerAcc> &conv_acc_impl);

    static void CreateImpFP(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs, LayerParam *param,
                            std::shared_ptr<ArmLayerAcc> &conv_acc_impl);

    static void CreateImpHalf(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs, LayerParam *param,
                              std::shared_ptr<ArmLayerAcc> &conv_acc_impl);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_CONV_LAYER_GROUP_H_
