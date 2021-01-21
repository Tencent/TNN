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

#ifndef TNN_SOURCE_TNN_DEVICE_X86_X86_UNARY2_LAYER_ACC_H
#define TNN_SOURCE_TNN_DEVICE_X86_X86_UNARY2_LAYER_ACC_H

#include <algorithm>
#include "tnn/device/x86/acc/Float4.h"
#include "tnn/device/x86/acc/Float8.h"
#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

typedef struct x86_unary2_operator {
public:
    virtual Status Init(LayerParam *param = nullptr) {
        param_ = param;
        return TNN_OK;
    }

    virtual float operator()(const float v) {
        return v;
    }

    virtual Float4 operator()(const Float4 &v) {
        return v;
    }

    virtual Float8 operator()(const Float8 &v) {
        return v;
    }

protected:
    LayerParam *param_ = nullptr;
} X86_UNARY2_OP;

template <typename UNARY2_OP>
void unary2_kernel_avx(std::vector<int> dims, const float *src, float *dst, LayerParam *param) {
    UNARY2_OP op;
    op.Init(param);

    auto count = DimsVectorUtils::Count(dims);
    int x      = 0;
    for (; x + 7 < count; x += 8) {
        Float8::saveu(dst + x, op(Float8::loadu(src + x)));
    }
    for (; x < count; x++) {
        dst[x] = op(src[x]);
    }
}

template <typename UNARY2_OP>
void unary2_kernel_sse(std::vector<int> dims, const float *src, float *dst, LayerParam *param) {
    UNARY2_OP op;
    op.Init(param);

    auto count = DimsVectorUtils::Count(dims);

    int x = 0;
    for (; x + 3 < count; x += 4) {
        Float4::save(dst + x, op(Float4::load(src + x)));
    }
    for (; x < count; x++) {
        dst[x] = op(src[x]);
    }
}

using unary2_kernel_avx_func_t = decltype(&unary2_kernel_avx<X86_UNARY2_OP>);
using unary2_kernel_sse_func_t = decltype(&unary2_kernel_sse<X86_UNARY2_OP>);

class X86Unary2LayerAcc : public X86LayerAcc {
public:
    virtual ~X86Unary2LayerAcc();

    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    static Status RegisterUnary2Kernel(LayerType type, x86_isa_t arch, unary2_kernel_avx_func_t kernel);
    static Status GetUnary2Kernel(LayerType type, x86_isa_t arch, unary2_kernel_avx_func_t &kernel);

protected:
    // std::shared_ptr<X86_UNARY2_OP> op_;
    LayerType type_;

    static std::string GetUnaryKernelName(LayerType type, x86_isa_t arch);
    static std::map<std::string, unary2_kernel_avx_func_t> &GetUnary2KernelMap();
};

class X86Unary2KernelRegister {
public:
    explicit X86Unary2KernelRegister(LayerType type, x86_isa_t arch, unary2_kernel_avx_func_t kernel) {
        X86Unary2LayerAcc::RegisterUnary2Kernel(type, arch, kernel);
    }
};

#define X86_REGISTER_UNARY2_KERNEL(layer_type, arch, kernel)                                                           \
    X86Unary2KernelRegister g_x86_##layer_type##_##arch##_unary2_register(layer_type, arch, kernel);

#define DECLARE_X86_UNARY2_ACC(type_string, op_type)                                                                   \
    class X86##type_string##LayerAcc : public X86Unary2LayerAcc {                                                      \
    public:                                                                                                            \
        X86##type_string##LayerAcc() {                                                                                 \
            type_ = op_type;                                                                                           \
        };                                                                                                             \
        virtual ~X86##type_string##LayerAcc(){};                                                                       \
    }
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_X86_X86_UNARY2_LAYER_ACC_H