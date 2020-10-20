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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_DECONV_LAYER_STRIDE_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_DECONV_LAYER_STRIDE_H_

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/arm_device.h"
#include "tnn/interpreter/layer_resource.h"

namespace TNN_NS {

class ArmDeconvLayerStride : public ArmLayerAcc {
public:
    virtual ~ArmDeconvLayerStride();

    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    static bool isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs);

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    struct ConvUnit {
        int kc_x;
        int kc_y;
        int x_offset;
        int y_offset;
        std::shared_ptr<ConvLayerParam> param;
        std::shared_ptr<ConvLayerResource> resource;
        std::shared_ptr<ArmLayerAcc> conv_acc_impl;
        std::shared_ptr<Blob> blob;
    };

private:
    Status CreateStrideConvUnit();

    Status SplitResource();

    Status SetSplitBlobDesc(Blob *blob);
    Status SetSplitBlobHandle(Blob *blob, RawBuffer &buf);

    Status CopyOutputSplitBlob(Blob *output);

    template <typename T>
    void CopyWithStride(ConvUnit &unit, Blob* output);

private:
    std::vector<ConvUnit> conv_units_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_DECONV_LAYER_STRIDE_H_
