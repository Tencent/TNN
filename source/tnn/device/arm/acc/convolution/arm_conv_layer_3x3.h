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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_CONV_LAYER_ACC_3X3_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_CONV_LAYER_ACC_3X3_H_

#include "tnn/device/arm/acc/compute/winograd_function.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_common.h"
#include "tnn/utils/winograd_generator.h"

namespace TNN_NS {

class ArmConvLayer3x3 : public ArmConvLayerCommon {
public:
    virtual ~ArmConvLayer3x3();

    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    template <typename T>
    Status Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    static bool isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs);

    static int SelectWinograd(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                              const std::vector<Blob *> &outputs);
                              
    virtual Status allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    int src_unit_;
    int dst_unit_;
    SrcTransformFunc SrcTransformFunc_;
    DstTransformFunc DstTransformFunc_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_CONV_LAYER_ACC_3X3_H_
