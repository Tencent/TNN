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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_GRADIENT_ARM_BINARY_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_GRADIENT_ARM_BINARY_LAYER_ACC_H_

#include "tnn/device/arm/acc/gradient/arm_gradient_layer_acc.h"

namespace TNN_NS {

typedef struct arm_binary_grad_function {
public:
    virtual std::pair<float, float> operator()(const float &i_0, const float &i_1, const float &o, const float &og) {
        return {og, og};
    }
    virtual std::pair<Float4, Float4> operator()(const Float4 &i_0, const Float4 &i_1, const Float4 &o,
                                                 const Float4 &og) {
        return {og, og};
    }
} ARM_BINARY_GRAD_FUNC;

#define DEFINE_ARM_BINARY_LAYER_GRAD_FUNC(grad_class, func_type)                                                       \
    template <int acc_0, int acc_1>                                                                                    \
    void ExecGrad(int count_quad, float *input_grad_0, float *input_grad_1, float *input_0, float *input_1,            \
                  float *output, float *output_grad) {                                                                 \
        Float4 in_0, in_1, in_grad_0, in_grad_1, out, out_grad;                                                        \
        OMP_PARALLEL_FOR_                                                                                              \
        for (int n = 0; n < count_quad; ++n) {                                                                         \
            in_0          = Float4::load(input_0 + n * 4);                                                             \
            in_1          = Float4::load(input_1 + n * 4);                                                             \
            out           = Float4::load(output + n * 4);                                                              \
            out_grad      = Float4::load(output_grad + n * 4);                                                         \
            auto in_grads = func_type()(in_0, in_1, out, out_grad);                                                    \
            in_grad_0     = in_grads.first;                                                                            \
            in_grad_1     = in_grads.second;                                                                           \
            Float4::save(input_grad_0 + n * 4, acc_0 ? (in_grad_0 + Float4::load(input_grad_0 + n * 4)) : in_grad_0);  \
            Float4::save(input_grad_1 + n * 4, acc_1 ? (in_grad_1 + Float4::load(input_grad_1 + n * 4)) : in_grad_1);  \
        }                                                                                                              \
    }

#define DEFINE_ARM_BINARY_GRAD_OP(type_string, func_type)                                                              \
    class Arm##type_string##GradOp : public GradOp {                                                                   \
    public:                                                                                                            \
        virtual ~Arm##type_string##GradOp(){};                                                                         \
        virtual Status OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,                   \
                              LayerResource *resource, LayerParam *param, Context *context,                            \
                              const GradOpInfo &grad_info) {                                                           \
            ON_GRAD_PREPARATION_IOR(2, 1, 0);                                                                          \
            if (!DimsVectorUtils::Equal(input_dims[0], input_dims[1])) {                                               \
                return Status(TNNERR_TRAIN_ERROR, "Arm" #type_string "GradOp input0 and input1 dims not match");       \
            }                                                                                                          \
            if (!DimsVectorUtils::Equal(input_dims[0], output_dims[0])) {                                              \
                return Status(TNNERR_TRAIN_ERROR, "Arm" #type_string "GradOp input and output dims not match");        \
            }                                                                                                          \
            int batch   = DimsFunctionUtils::GetDim(input_dims[0], 0);                                                 \
            int channel = DimsFunctionUtils::GetDim(input_dims[0], 1);                                                 \
            int hw      = DimsVectorUtils::Count(input_dims[0], 2);                                                    \
            if (fw_inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {                                            \
                int count_quad        = batch * UP_DIV(channel, 4) * hw;                                               \
                auto input_ptr_0      = reinterpret_cast<float *>(GetBlobHandlePtr(fw_inputs[0]->GetHandle()));        \
                auto input_ptr_1      = reinterpret_cast<float *>(GetBlobHandlePtr(fw_inputs[1]->GetHandle()));        \
                auto output_ptr       = reinterpret_cast<float *>(GetBlobHandlePtr(fw_outputs[0]->GetHandle()));       \
                auto input_grad_ptr_0 = reinterpret_cast<float *>(GetBlobHandlePtr(input_grads[0]->GetHandle()));      \
                auto input_grad_ptr_1 = reinterpret_cast<float *>(GetBlobHandlePtr(input_grads[1]->GetHandle()));      \
                auto output_grad_ptr  = reinterpret_cast<float *>(GetBlobHandlePtr(output_grads[0]->GetHandle()));     \
                if (acc_input_grads[0]) {                                                                              \
                    if (acc_input_grads[1]) {                                                                          \
                        ExecGrad<1, 1>(count_quad, input_grad_ptr_0, input_grad_ptr_1, input_ptr_0, input_ptr_1,       \
                                       output_ptr, output_grad_ptr);                                                   \
                    } else {                                                                                           \
                        ExecGrad<1, 0>(count_quad, input_grad_ptr_0, input_grad_ptr_1, input_ptr_0, input_ptr_1,       \
                                       output_ptr, output_grad_ptr);                                                   \
                    }                                                                                                  \
                } else {                                                                                               \
                    if (acc_input_grads[1]) {                                                                          \
                        ExecGrad<0, 1>(count_quad, input_grad_ptr_0, input_grad_ptr_1, input_ptr_0, input_ptr_1,       \
                                       output_ptr, output_grad_ptr);                                                   \
                    } else {                                                                                           \
                        ExecGrad<0, 0>(count_quad, input_grad_ptr_0, input_grad_ptr_1, input_ptr_0, input_ptr_1,       \
                                       output_ptr, output_grad_ptr);                                                   \
                    }                                                                                                  \
                }                                                                                                      \
            } else {                                                                                                   \
                LOGE("Arm" #type_string "GradOp::OnGrad, dtype not supported\n");                                      \
                return Status(TNNERR_TRAIN_ERROR, "dtype not supported");                                              \
            }                                                                                                          \
            return TNN_OK;                                                                                             \
        }                                                                                                              \
                                                                                                                       \
    private:                                                                                                           \
        DEFINE_ARM_BINARY_LAYER_GRAD_FUNC(Arm##type_string##GradOp, func_type)                                         \
    };

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_GRADIENT_ARM_BINARY_LAYER_ACC_H_
