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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_UNARY_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_UNARY_LAYER_ACC_H_

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/arm_common.h"

namespace TNN_NS {

typedef struct arm_unary_operator {
public:
    virtual Status Init(LayerParam *param = nullptr) {
        param_ = param;
        return TNN_OK;
    }

    virtual float operator()(const float &v) {
        return v;
    }

    virtual Float4 operator()(const Float4 &v) {
        return v;
    };
    virtual Float4 fast_op(const Float4 &v) {
        return operator()(v);
    };

    virtual fp16_t operator()(const fp16_t &v) {
        return v;
    }

protected:
    LayerParam *param_ = nullptr;
} ARM_UNARY_OP;

class ArmUnaryLayerAcc : public ArmLayerAcc {
public:
    virtual ~ArmUnaryLayerAcc();

    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

protected:
    virtual bool DataTypeSupported(DataType data_type) override;

    std::shared_ptr<ARM_UNARY_OP> op_;

private:
    template <typename T>
    Status Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
};

#define DECLARE_ARM_UNARY_ACC(type_string, op_type)                                                                    \
    class Arm##type_string##LayerAcc : public ArmUnaryLayerAcc {                                                       \
    public:                                                                                                            \
        Arm##type_string##LayerAcc() {                                                                                 \
            op_ = std::make_shared<op_type>();                                                                         \
        }                                                                                                              \
        virtual ~Arm##type_string##LayerAcc(){};                                                                       \
    }

#if TNN_ARM82
#define DEFINE_ARM_FP16_UNARY_DO_FORWARD                                                                               \
    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {                  \
        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {                                                    \
            return ExecFp16(inputs, outputs);                                                                          \
        } else {                                                                                                       \
            return ArmUnaryLayerAcc::DoForward(inputs, outputs);                                                       \
        }                                                                                                              \
    }
#else
#define DEFINE_ARM_FP16_UNARY_DO_FORWARD
#endif  // TNN_ARM82

#define DECLARE_ARM_UNARY_ACC_FP16(type_string, op_type)                                                               \
    class Arm##type_string##LayerAcc : public ArmUnaryLayerAcc {                                                       \
    public:                                                                                                            \
        Arm##type_string##LayerAcc() {                                                                                 \
            op_ = std::make_shared<op_type>();                                                                         \
        }                                                                                                              \
        DEFINE_ARM_FP16_UNARY_DO_FORWARD;                                                                              \
        virtual ~Arm##type_string##LayerAcc(){};                                                                       \
                                                                                                                       \
    private:                                                                                                           \
        DECLARE_ARM_FP16_LAYER_FUNC;                                                                                   \
    }

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_UNARY_LAYER_ACC_H_
