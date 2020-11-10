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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_CONV_LAYER_GROUP_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_CONV_LAYER_GROUP_H_

#include "tnn/device/arm/acc/convolution/arm_conv_layer_acc_factory.h"
#include "tnn/device/arm/arm_device.h"
#include "tnn/interpreter/layer_resource.h"

namespace TNN_NS {

class ArmConvLayerGroup : public ArmLayerAcc {
public:
    virtual ~ArmConvLayerGroup();

    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

private:
    Status SetGroupParam(std::shared_ptr<LayerParam> &group_param);

    Status SplitResource(std::vector<std::shared_ptr<LayerResource>> &resources);

    Status SetSplitBlobDesc(Blob *blob, std::vector<std::shared_ptr<Blob>> &blobs);
    Status SetSplitBlobHandle(std::vector<std::shared_ptr<Blob>> &blobs, RawBuffer &buf);
    Status SetSplitBlobScale(Blob *blob, std::vector<std::shared_ptr<Blob>> &blobs);

    Status CopyInputSplitBlob(Blob *input);
    Status CopyOutputSplitBlob(Blob *output);

    template <typename T>
    void TransformInput(Blob *input);
    template <typename T>
    void TransformOutput(Blob *input);

private:
    std::vector<std::shared_ptr<ArmLayerAcc>> conv_acc_impls_;
    std::vector<std::shared_ptr<Blob>> group_inputs_;
    std::vector<std::shared_ptr<Blob>> group_outputs_;

    std::shared_ptr<LayerParam> group_conv_param_ = nullptr;
    std::vector<std::shared_ptr<IntScaleResource>> group_scale_res_;

    int group_ = 1;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_CONV_LAYER_GROUP_H_
