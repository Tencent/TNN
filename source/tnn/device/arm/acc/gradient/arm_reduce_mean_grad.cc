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

#include "tnn/device/arm/acc/gradient/arm_gradient_layer_acc.h"

namespace TNN_NS {

DECLARE_ARM_GRAD_OP(ReduceMean, LAYER_REDUCE_MEAN);

Status ArmReduceMeanGradOp::OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                      LayerResource *resource, GradientParam *grad_param, Context *context,
                                      const GradOpInfo &grad_info) {
    ON_GRAD_PREPARATION_IOR(1, 1, 0);

    int input_count  = DimsVectorUtils::Count(input_dims[0]);
    int output_count = DimsVectorUtils::Count(output_dims[0]);
    float ratio      = float(output_count) / float(input_count);

    if (output_count != 1) {
        LOGE("ArmReduceMeanGradOp::OnGrad, only all reduce supported yet, but output count is %d\n", output_count);
        return Status(TNNERR_LAYER_ERR, "only all reduce supported yet");
    }

    int batch   = DimsFunctionUtils::GetDim(input_dims[0], 0);
    int channel = DimsFunctionUtils::GetDim(input_dims[0], 1);
    int hw      = DimsVectorUtils::Count(input_dims[0], 2);

    if (fw_inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        int count_quad = batch * UP_DIV(channel, 4) * hw;

        Float4 grad = Float4(ratio);

        float *output_grad_ptr = (float *)GetBlobHandlePtr(output_grads[0]->GetHandle());
        grad                   = grad * (*output_grad_ptr);

        auto input_grad_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(input_grads[0]->GetHandle()));

        if (!acc_input_grads[0]) {
            OMP_PARALLEL_FOR_
            for (int n = 0; n < count_quad; n++) {
                Float4::save(input_grad_ptr + n * 4, grad);
            }
        } else {
            OMP_PARALLEL_FOR_
            for (int n = 0; n < count_quad; n++) {
                Float4::save(input_grad_ptr + n * 4, grad + Float4::load(input_grad_ptr + n * 4));
            }
        }
    } else {
        LOGE("ArmReduceMeanGradOp::OnGrad, dtype not supported\n");
        return Status(TNNERR_LAYER_ERR, "dtype not supported");
    }

    return TNN_OK;
}

REGISTER_ARM_GRAD_OP(ReduceMean, LAYER_REDUCE_MEAN)
REGISTER_ARM_GRAD_LAYOUT(LAYER_REDUCE_MEAN, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
