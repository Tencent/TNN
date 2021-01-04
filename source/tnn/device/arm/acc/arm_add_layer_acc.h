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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_Add_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_Add_LAYER_ACC_H_

#include "tnn/device/arm/acc/arm_layer_acc.h"

namespace TNN_NS {

enum AddOpType { ADD_SINGLE = 1, ADD_CHANNEL = 2, ADD_ELEMENT = 3 };

#define OperatorAddPreparation()                                      \
    DimsVector dims_broadcast;                                        \
    if (DimsVectorUtils::Equal(dims0, dims1, 2)) {                    \
        dims_broadcast.clear();                                       \
        type = ADD_ELEMENT;                                           \
        if (dims0[0] != dims[0] || dims0[1] != dims[1])               \
            std::swap(_input0, _input1);                              \
    } else if (DimsVectorUtils::Equal(dims0, dims, 1)) {              \
        dims_broadcast = dims1;                                       \
    } else {                                                          \
        dims_broadcast = dims0;                                       \
        std::swap(_input0, _input1);                                  \
    }                                                                 \
    if (dims_broadcast.size()) {                                      \
        type = (dims_broadcast[1] == 1) ? ADD_SINGLE : ADD_CHANNEL;   \
    }

// @brief conv layer cpu acc
class ArmAddLayerAcc : public ArmLayerAcc {
public:
    virtual ~ArmAddLayerAcc();

    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    Status allocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    
    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

private:
    RawBuffer input0_int_scale_;
    RawBuffer input1_int_scale_;
    RawBuffer output_int_scale_;
    RawBuffer output_bias_;
    DimsVector bias_shape_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_Add_LAYER_ACC_H_
