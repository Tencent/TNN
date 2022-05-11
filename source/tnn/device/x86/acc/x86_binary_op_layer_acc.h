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

#ifndef TNN_SOURCE_TNN_DEVICE_X86_ACC_X86_BINARY_OP_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_X86_ACC_X86_BINARY_OP_LAYER_ACC_H_

#include <vector>

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/x86_device.h"
#include "tnn/device/x86/acc/Float4.h"
#include "tnn/interpreter/layer_resource_generator.h"

namespace TNN_NS {

enum class X86BinaryOpType : int {
    kADD = 0,
    kSUB = 1,
    kMUL = 2,
    kDIV = 3,
    kMAX = 4,
    kMIN = 5,
};

template <X86BinaryOpType op_type, typename VEC, int pack>
Status BinaryFunc(float *output_ptr, const float *input0_ptr, const float *input1_ptr,
                  DimsVector &dims0, DimsVector &dims1, DimsVector &output_dims);
template <X86BinaryOpType op_type>
void BinaryGeneral(DimsVector output_shape, const std::vector<DimsVector> &input_shapes,
                   float *output_ptr, std::vector<float *> &input_ptrs);
using binary_func_t = decltype(&BinaryFunc<X86BinaryOpType::kADD, Float4, 4>);
using binary_general_func_t = decltype(&BinaryGeneral<X86BinaryOpType::kADD>);

class X86BinaryOpLayerAcc : public X86LayerAcc {
public:
    virtual ~X86BinaryOpLayerAcc();

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,
                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;
protected:
    // Calculate Function
    Status Calculate(const std::vector<Blob *> &input_blobs, const std::vector<void *> &input_ptrs,
                     const std::vector<DimsVector> &input_shapes, Blob *output);
    X86BinaryOpType op_type_;
private:
    std::vector<DimsVector> input_shapes_;
    BroadcastType btype_;

    binary_func_t binary_func_;
    binary_general_func_t binary_general_func_;

    std::shared_ptr<LayerResource> binary_acc_f32_resource_ = nullptr;
};

#define DECLARE_X86_BINARY_OP_ACC(type_string, op_type)                                                                 \
    class X86##type_string##LayerAcc : public X86BinaryOpLayerAcc {                                                     \
    public:                                                                                                             \
        X86##type_string##LayerAcc() {                                                                                  \
            X86BinaryOpLayerAcc::op_type_ = op_type;                                                                    \
        }                                                                                                               \
        virtual ~X86##type_string##LayerAcc(){};                                                                        \
                                                                                                                        \
    }
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_X86_ACC_X86_BINARY_OP_LAYER_ACC_H_
