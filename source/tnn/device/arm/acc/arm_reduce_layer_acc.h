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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_REDUCE_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_REDUCE_LAYER_ACC_H_

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

typedef struct arm_reduce_operator {
public:
    virtual void DataInit(void *data, size_t count) {
        memset(data, 0, count * sizeof(float));
    };

    virtual Float4 DataInit() {
        return Float4(0);
    };

    virtual Float4 PreCalculate(Float4 &v) {
        return v;
    };

    virtual float PreCalculate(const float &v) {
        return v;
    };

    virtual Float4 Calculate(Float4 &v, Float4 &t) {
        return v + t;
    };

    virtual float Calculate(const float &v, const float &t) {
        return v + t;
    };

    virtual Float4 PostCalculate(const Float4 &v, const Float4 &t) {
        return v;
    };

    virtual float PostCalculate(const float &v, const float &t) {
        return v;
    };

    virtual bool NeedPreCalculate() {
        return false;
    };

    virtual bool PosCalculateOnce() {
        return false;
    };

} ARM_REDUCE_OP;

class ArmReduceLayerAcc : public ArmLayerAcc {
public:
    virtual ~ArmReduceLayerAcc();

    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

protected:
    std::shared_ptr<ARM_REDUCE_OP> op_;
    template <bool post_cal>
    void ReduceOneAxis(float* input_data, float* output_data, DimsVector& dims_in, int out_count, int axis);
    template <bool post_cal>
    void ReduceChannel(float* input_data, float* output_data, DimsVector& dims_in,
        const int c4n, const int c4r, const Float4 axis_n, const int hw_r, const int hw_c, const int hw);
};

#define DECLARE_ARM_REDUCE_ACC(type_string, op_type)                                                                   \
    class Arm##type_string##LayerAcc : public ArmReduceLayerAcc {                                                      \
    public:                                                                                                            \
        Arm##type_string##LayerAcc() {                                                                                 \
            op_ = std::make_shared<op_type>();                                                                         \
        }                                                                                                              \
        virtual ~Arm##type_string##LayerAcc(){};                                                                       \
    }
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_REDUCE_LAYER_ACC_H_
