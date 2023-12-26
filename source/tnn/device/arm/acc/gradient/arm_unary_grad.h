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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_GRADIENT_ARM_UNARY_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_GRADIENT_ARM_UNARY_LAYER_ACC_H_

#include "tnn/device/arm/acc/gradient/arm_gradient_layer_acc.h"

namespace TNN_NS {

class ArmUnaryGradOp : public GradOp {
public:
    virtual ~ArmUnaryGradOp(){};
    virtual Status OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                          LayerResource *resource, GradientParam *grad_param, Context *context,
                          const GradOpInfo &grad_info) {
        ON_GRAD_PREPARATION_IOR(1, 1, 0);
        if (!DimsVectorUtils::Equal(input_dims[0], output_dims[0])) {
            return Status(TNNERR_TRAIN_ERROR, "ArmUnaryGradOp input and output dims not match");
        }
        int batch   = DimsFunctionUtils::GetDim(input_dims[0], 0);
        int channel = DimsFunctionUtils::GetDim(input_dims[0], 1);
        int hw      = DimsVectorUtils::Count(input_dims[0], 2);
        if (fw_inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            int count_quad       = batch * UP_DIV(channel, 4) * hw;
            auto input_ptr       = reinterpret_cast<float *>(GetBlobHandlePtr(fw_inputs[0]->GetHandle()));
            auto output_ptr      = reinterpret_cast<float *>(GetBlobHandlePtr(fw_outputs[0]->GetHandle()));
            auto input_grad_ptr  = reinterpret_cast<float *>(GetBlobHandlePtr(input_grads[0]->GetHandle()));
            auto output_grad_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(output_grads[0]->GetHandle()));
            ExecGrad(count_quad, input_grad_ptr, input_ptr, output_ptr, output_grad_ptr);
        } else {
            LOGE("ArmUnaryGradOp::OnGrad, dtype not supported\n");
            return Status(TNNERR_TRAIN_ERROR, "dtype not supported");
        }
        return TNN_OK;
    }

private:
    virtual float cal_grad(const float &i, const float &o, const float &og)     = 0;
    virtual Float4 cal_grad(const Float4 &i, const Float4 &o, const Float4 &og) = 0;

    void ExecGrad(int count_quad, float *input_grad, float *input, float *output, float *output_grad) {
        OMP_PARALLEL_FOR_
        for (int n = 0; n < count_quad; ++n) {
            Float4 in       = Float4::load(input + n * 4);
            Float4 out      = Float4::load(output + n * 4);
            Float4 out_grad = Float4::load(output_grad + n * 4);
            auto in_grad    = cal_grad(in, out, out_grad);
            Float4::save(input_grad + n * 4, in_grad + Float4::load(input_grad + n * 4));
        }
    }
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_GRADIENT_ARM_UNARY_LAYER_ACC_H_
