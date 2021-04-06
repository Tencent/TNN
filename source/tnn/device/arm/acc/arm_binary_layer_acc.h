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
enum class ArmBinaryOpType : int {
    kADD = 0,
    kSUB = 1,
    kMUL = 2,
    kDIV = 3,
    kMAX = 4,
    kMIN = 5,
    kHARDSWISH = 6,
};

class ArmBinaryLayerAcc : public ArmLayerAcc {
public:
    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs) override;

    virtual ~ArmBinaryLayerAcc();

    Status allocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    template <typename T, ArmBinaryOpType op_type>
    Status Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    // int8 will be implemented inside op
    virtual Status ExecInt8(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

#if TNN_ARM82
    virtual Status allocateBufferParamHalf(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    template <ArmBinaryOpType op_type>
    Status ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
#endif

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

protected:
    virtual bool DataTypeSupported(DataType data_type) override;
    virtual Status ConfigBuffer2ArmBlobDesc(BlobDesc &desc) override;

    ArmBinaryOpType op_type_;
    // used for hardswish
    float alpha_ = 0.f;
    float beta_ = 0.f;
private:
    RawBuffer broadcast_;

    std::vector<void *> input_ptrs_;
    std::vector<DimsVector> input_shapes_;
    BroadcastType btype_;
    BlobDesc desc_for_config_const_blob_;
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
