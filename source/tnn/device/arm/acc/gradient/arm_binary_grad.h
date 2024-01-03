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

#include "tnn/device/arm/acc/arm_binary_layer_acc.h"
#include "tnn/device/arm/acc/gradient/arm_gradient_layer_acc.h"
#include "tnn/layer/multidir_broadcast_layer.h"

namespace TNN_NS {

class ArmBinaryGradOp : public GradOp {
public:
    virtual ~ArmBinaryGradOp(){};
    virtual Status OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                          LayerResource *resource, GradientParam *grad_param, Context *context,
                          const GradOpInfo &grad_info) {
        ON_GRAD_PREPARATION()

        auto forward_acc = dynamic_cast<ArmBinaryLayerAcc *>(forward_base->GetAbstractLayerAcc());
        CHECK_PARAM_NULL(forward_acc);

        auto forward_param = dynamic_cast<MultidirBroadcastLayerParam *>(grad_param->forward_param);
        CHECK_PARAM_NULL(forward_param);

        // 求导目标  dL/dx + dL/dw
        void *grad_ptrs[2];
        if (forward_acc->GetResource().GetDataCount() > 0) {  // 如果w不需要训练，则w_grad设置为nullptr
            if (forward_param->weight_input_index == 0) {     // w 作为左值
                grad_ptrs[0] = resource_grads.empty() ? nullptr : resource_grads[0]->GetHandle().force_to<void *>();
                grad_ptrs[1] = input_grads[0]->GetHandle().force_to<void *>();
            } else {  // w 作为右值
                grad_ptrs[0] = input_grads[0]->GetHandle().force_to<void *>();
                grad_ptrs[1] = resource_grads.empty() ? nullptr : resource_grads[0]->GetHandle().force_to<void *>();
            }
        } else {
            grad_ptrs[0] = input_grads[0]->GetHandle().force_to<void *>();
            grad_ptrs[1] = input_grads[1]->GetHandle().force_to<void *>();
        }

        // 前向的输入：x+w
        std::vector<void *> &forward_input_ptrs     = forward_acc->GetInputPtrs();
        std::vector<DimsVector> &forward_input_dims = forward_acc->GetInputShapes();

        // 目前只处理float
        if (fw_inputs[0]->GetBlobDesc().data_type != DATA_TYPE_FLOAT) {
            LOGE("Arm Binary GradOp::OnGrad, dtype not supported\n");
            return Status(TNNERR_TRAIN_ERROR, "dtype not supported");
        }

        if (fw_inputs[0]->GetBlobDesc().data_format != DATA_FORMAT_NC4HW4) {
            LOGE("Arm Binary GradOp::OnGrad, data format not supported\n");
            return Status(TNNERR_TRAIN_ERROR, "data format not supported");
        }

        float *x0      = reinterpret_cast<float *>(forward_input_ptrs[0]);
        float *x1      = reinterpret_cast<float *>(forward_input_ptrs[1]);
        float *y       = reinterpret_cast<float *>(GetBlobHandlePtr(fw_outputs[0]->GetHandle()));
        float *y_grad  = reinterpret_cast<float *>(GetBlobHandlePtr(output_grads[0]->GetHandle()));
        float *x0_grad = reinterpret_cast<float *>(grad_ptrs[0]);
        float *x1_grad = reinterpret_cast<float *>(grad_ptrs[1]);
        if (DimsVectorUtils::Equal(forward_input_dims[0], forward_input_dims[1])) {  // BroadcastTypeNormal
            ExecGradNormally(x0_grad, x1_grad, x0, x1, y, y_grad, fw_outputs[0]->GetBlobDesc().dims);
        } else if (DimsVectorUtils::Count(forward_input_dims[0]) == 1 ||
                   DimsVectorUtils::Count(forward_input_dims[1]) == 1) {  // BroadcastTypeSingle
            if (DimsVectorUtils::Count(forward_input_dims[1]) == 1) {     // 确保x0是单个元素，需要被扩展
                std::swap(x0, x1);
                std::swap(x0_grad, x1_grad);
            }
            ExecGradSingle(x0_grad, x1_grad, x0, x1, y, y_grad, fw_outputs[0]->GetBlobDesc().dims);
        } else {
            LOGE("Arm Mul GradOp::OnGrad, broadcast type not supported\n");
            return Status(TNNERR_TRAIN_ERROR, "broadcast type not supported");
        }
        return TNN_OK;
    }

private:
    virtual std::pair<float, float> cal_grad(const float &i_0, const float &i_1, const float &o, const float &og) = 0;
    virtual std::pair<Float4, Float4> cal_grad(const Float4 &i_0, const Float4 &i_1, const Float4 &o,
                                               const Float4 &og)                                                  = 0;

    void ExecGradNormally(float *x0_grad, float *x1_grad, float *x0, float *x1, float *y, float *y_grad,
                          const DimsVector &dims) {
        int count_quad = DimsFunctionUtils::GetNCHWXPackedCount(dims, 4);
        OMP_PARALLEL_FOR_
        for (int n = 0; n < count_quad; ++n) {
            Float4 in_0     = Float4::load(x0 + n * 4);
            Float4 in_1     = Float4::load(x1 + n * 4);
            Float4 out      = Float4::load(y + n * 4);
            Float4 out_grad = Float4::load(y_grad + n * 4);
            auto in_grads   = cal_grad(in_0, in_1, out, out_grad);
            auto &in_grad_0 = in_grads.first;
            auto &in_grad_1 = in_grads.second;
            if (x0_grad != nullptr) {
                Float4::save(x0_grad + n * 4, in_grad_0 + Float4::load(x0_grad + n * 4));
            }
            if (x1_grad != nullptr) {
                Float4::save(x1_grad + n * 4, in_grad_1 + Float4::load(x1_grad + n * 4));
            }
        }
    }

    void ExecGradSingle(float *x0_grad, float *x1_grad, float *x0, float *x1, float *y, float *y_grad,
                        const DimsVector &dims) {
        int batch = DimsFunctionUtils::GetDim(dims, 0);
        int c     = DimsFunctionUtils::GetDim(dims, 1);
        int c4n   = UP_DIV(c, 4);
        int c4r   = DimsFunctionUtils::GetDim(dims, 1) % 4;
        int hw    = DimsVectorUtils::Count(dims, 2);

        int nptr = 0;
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < c4n; c++) {
                OMP_PARALLEL_FOR_
                for (int i = 0; i < hw; i++) {
                    auto in_0     = Float4(x0[0]);
                    auto in_1     = Float4::load(x1 + nptr);
                    auto out      = Float4::load(y + nptr);
                    auto out_grad = Float4::load(y_grad + nptr);
                    auto grads    = cal_grad(in_0, in_1, out, out_grad);
                    auto &grad_0  = grads.first;
                    auto &grad_1  = grads.second;

                    // x0的grad需要累加
                    if (x0_grad != nullptr) {
                        int e = 4;
                        if ((c == c4n - 1) && (c4r != 0)) {
                            e = c4r;
                        }
                        float grad_0_sum = 0.0;
                        for (int k = 0; k < e; ++k) {
                            grad_0_sum += grad_0[k];
                        }
                        x0_grad[0] += grad_0_sum;
                    }
                    if (x1_grad != nullptr) {
                        Float4::save(x1_grad + nptr, grad_1 + Float4::load(x1_grad + nptr));
                    }

                    nptr += 4;
                }
            }
        }
    }
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_GRADIENT_ARM_BINARY_LAYER_ACC_H_
