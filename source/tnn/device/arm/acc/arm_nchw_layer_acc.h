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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_NCHW_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_NCHW_LAYER_ACC_H_

#include "tnn/device/arm/acc/arm_layer_acc.h"

namespace TNN_NS {

// @brief NCHW layer cpu acc
class ArmNchwLayerAcc : public ArmLayerAcc {
public:
    virtual ~ArmNchwLayerAcc();

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    virtual Status AllocConvertBuffer(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    template <typename T>
    Status UnPackInputs(const std::vector<Blob *> &inputs);

    template <typename T>
    Status PackOutputs(const std::vector<Blob *> &outputs);

    std::vector<Blob *> GetNchwBlobVector(const std::vector<std::shared_ptr<Blob>> &blobs);

    std::vector<std::shared_ptr<Blob>> nchw_blob_in;
    std::vector<std::shared_ptr<Blob>> nchw_blob_out;
};

}  // namespace TNN_NS

#define DECLARE_ARM_NCHW_ACC(type_string, layer_type)                                                                  \
    class Arm##type_string##LayerAcc : public ArmNchwLayerAcc {                                                        \
    public:                                                                                                            \
        virtual ~Arm##type_string##LayerAcc(){};                                                                       \
        virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);               \
    }

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_NCHW_LAYER_ACC_H_
