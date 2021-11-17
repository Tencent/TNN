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

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/train/gradient/layer_grad.h"
#include "tnn/utils/dims_function_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

DECLARE_ARM_LAYER_GRAD(Sigmoid, LAYER_SIGMOID);

// y = sigmoid(x)
// dy/dx = y(1-y)
template <int acc_g0>
static void ExecSigmoidGrad(int count_quad, float *output_ptr0, float *grad_ptr0, float *up_grad) {
    Float4 y0, g0, ug;
    OMP_PARALLEL_FOR_
    for (int n = 0; n < count_quad; ++n) {
        y0 = Float4::load(output_ptr0 + n * 4);
        g0 = y0 * (Float4(1.0) - y0);
        ug = Float4::load(up_grad + n * 4);
        g0 = g0 * ug;
        Float4::save(grad_ptr0 + n * 4, acc_g0 ? (g0 + Float4::load(grad_ptr0 + n * 4)) : g0);
    }
}

Status ArmSigmoidLayerGrad::OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                   LayerResource *resource, LayerParam *param, Context *context,
                                   LayerGradInfo *grad_info) {
    CHECK_PARAM_NULL(grad_info);
    if (grad_info->accumulate_blob_grad.size() < 1) {
        LOGD("ArmSigmoidLayerGrad::OnGrad, accumulate_blob_grad error\n");
        return Status(TNNERR_LAYER_ERR, "accumulate_blob_grad size error");
    }
    bool accumulate_blob_grad0 = grad_info->accumulate_blob_grad[0];

    if (inputs.size() != 3 || outputs.size() != 1) {
        return Status(TNNERR_LAYER_ERR, "input size or output size not match in ArmSigmoidLayerGrad");
    }

    auto fw_input0     = inputs[0];
    auto fw_output     = inputs[1];
    auto upstream_grad = inputs[2];

    auto input0_dims = fw_input0->GetBlobDesc().dims;
    auto output_dims = fw_output->GetBlobDesc().dims;
    if (!DimsVectorUtils::Equal(input0_dims, output_dims)) {
        return Status(TNNERR_LAYER_ERR, "ArmSigmoidLayerGrad input dims and output dims not match");
    }

    auto grad0 = outputs[0];
    auto dims0 = grad0->GetBlobDesc().dims;

    if (!DimsVectorUtils::Equal(input0_dims, dims0)) {
        return Status(TNNERR_LAYER_ERR, "ArmSigmoidLayerGrad input dims and grad dims not match");
    }

    int batch   = DimsFunctionUtils::GetDim(dims0, 0);
    int channel = DimsFunctionUtils::GetDim(dims0, 1);

    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        int count      = batch * ROUND_UP(channel, 4) * DimsVectorUtils::Count(dims0, 2);
        int count_quad = UP_DIV(count, 4);

        auto output_ptr        = reinterpret_cast<float *>(GetBlobHandlePtr(fw_output->GetHandle()));
        auto grad_ptr0         = reinterpret_cast<float *>(GetBlobHandlePtr(grad0->GetHandle()));
        auto upstream_grad_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(upstream_grad->GetHandle()));

        if (accumulate_blob_grad0) {
            ExecSigmoidGrad<1>(count_quad, output_ptr, grad_ptr0, upstream_grad_ptr);
        } else {
            ExecSigmoidGrad<0>(count_quad, output_ptr, grad_ptr0, upstream_grad_ptr);
        }
    } else {
        LOGE("ArmSigmoidLayerGrad::OnGrad, dtype not supported\n");
        return Status(TNNERR_LAYER_ERR, "dtype not supported");
    }

    return TNN_OK;
}

REGISTER_ARM_LAYER_GRAD(Sigmoid, LAYER_SIGMOID)
REGISTER_ARM_GRAD_LAYOUT(LAYER_SIGMOID, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
