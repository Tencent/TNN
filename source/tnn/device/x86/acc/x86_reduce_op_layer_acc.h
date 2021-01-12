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

#ifndef TNN_SOURCE_TNN_DEVICE_X86_ACC_X86_REDUCE_OP_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_X86_ACC_X86_REDUCE_OP_LAYER_ACC_H_

#include <vector>
#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/x86_device.h"

namespace TNN_NS {

enum class X86ReduceOpType : int {
    kL1     = 0,
    kL2     = 1,
    kMAX    = 2,
    kMIN    = 3,
    kMEAN   = 4,
    kSUM    = 5,
    kPROD   = 6,
    kSUMSQUARE  = 7,
    kLOGSUM     = 8,
    kLOGSUMEXP  = 9
};

class X86ReduceOpLayerAcc : public X86LayerAcc {
public:
    virtual ~X86ReduceOpLayerAcc();

    virtual Status Init(Context *context, LayerParam *param, LayerResource* resource, const std::vector<Blob*> &inputs,
                        const std::vector<Blob *> &outputs);
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    X86ReduceOpType op_type_;
};

#define DECLARE_X86_REDUCE_OP_ACC(type_string, op_type)                                                                 \
    class X86##type_string##LayerAcc : public X86ReduceOpLayerAcc {                                                     \
    public:                                                                                                             \
        X86##type_string##LayerAcc() {                                                                                  \
            X86ReduceOpLayerAcc::op_type_ = op_type;                                                                    \
        }                                                                                                               \
        virtual ~X86##type_string##LayerAcc(){};                                                                        \
                                                                                                                       \
    }
}  // namespace TNN_NS

#endif