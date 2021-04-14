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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_BINARY_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_BINARY_LAYER_ACC_H_

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/arm_common.h"

namespace TNN_NS {

// @brief conv layer cpu acc
class ArmBinaryLayerAcc : public ArmLayerAcc {
public:
    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs) override;

    virtual ~ArmBinaryLayerAcc();

    Status allocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    template <typename T>
    Status Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

protected:
    template <typename Tout, typename Tin1, typename Tin2>
    Status BinaryFunc(Tout *output_ptr, Tin1 *input0_ptr, Tin2 *input1_ptr, DimsVector &dims0, DimsVector &dims1);

    virtual bool DataTypeSupported(DataType data_type) override;
    
    std::function<Float4(const Float4 &v1, const Float4 &v2)> _Operator = nullptr;

private:
    RawBuffer broadcast_;
    RawBuffer input0_int_scale_;
    RawBuffer input1_int_scale_;
    RawBuffer output_int_scale_;
};

#define DECLARE_ARM_BINARY_ACC(type_string)                                                                            \
    class Arm##type_string##LayerAcc : public ArmBinaryLayerAcc {                                                      \
    public:                                                                                                            \
        virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,                              \
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;           \
        virtual ~Arm##type_string##LayerAcc() override;                                                                \
    }

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_Binary_LAYER_ACC_H_
