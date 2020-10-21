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

#include "immintrin.h"

namespace TNN_NS {

typedef struct x86_reduce_operator {
public:
#ifdef __AVX2__
    virtual __m256 PostProcess(const __m256 v_) { return v_; };
    virtual __m256 operator()(const __m256 v1_, const __m256 v2_) {};
#else
    virtual float PostProcess(const float v) { return v; };
    virtual float operator()(const float v1, const float v2) {};
#endif
} X86_REDUCE_OP;

class X86ReduceOpLayerAcc : public X86LayerAcc {
public:
    virtual ~X86ReduceOpLayerAcc();

    virtual Status Init(Context *context, LayerParam *param, LayerResource* resource, const std::vector<Blob*> &inputs,
                        const std::vector<Blob *> &outputs);
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    std::shared_ptr<X86_REDUCE_OP> op_;
};

#define DECLARE_X86_REDUCE_OP_ACC(type_string, op_type)                                                                 \
    class X86##type_string##LayerAcc : public X86ReduceOpLayerAcc {                                                     \
    public:                                                                                                             \
        X86##type_string##LayerAcc() {                                                                                  \
            X86ReduceOpLayerAcc::op_ = std::make_shared<op_type>();                                                     \
        }                                                                                                               \
        virtual ~X86##type_string##LayerAcc(){};                                                                        \
                                                                                                                       \
    }
}  // namespace TNN_NS

#endif