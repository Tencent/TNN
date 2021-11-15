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

DECLARE_ARM_LAYER_GRAD(BinaryCrossEntropy, LAYER_BINARY_CROSSENTROPY);

// x0 is logits, x1 is true labels
// y = -x1*log(x0) - (1-x1)*log(1-x0)
// dy/dx0 = (1-x1)/(1-x0) -x1/x0
// dy/dx1 = log(1-x0) - log(x0)
template <int acc_g0, int acc_g1>
static void ExecBCEGrad(int count_quad, float *input_ptr0, float *input_ptr1, float *grad_ptr0, float *grad_ptr1,
                        float *up_grad) {
    Float4 x0, x1, g0, g1, ug;
    OMP_PARALLEL_FOR_
    for (int n = 0; n < count_quad; ++n) {
        x0 = Float4::load(input_ptr0 + n * 4);
        x1 = Float4::load(input_ptr1 + n * 4);
        g0 = Float4::div(Float4(1.0) - x1, Float4(1.0) - x0) - Float4::div(x1, x0);
        g1 = Float4::log(Float4(1.0) - x0) - Float4::log(x0);
        ug = Float4::load(up_grad + n * 4);
        g0 = g0 * ug;
        g1 = g1 * ug;
        Float4::save(grad_ptr0 + n * 4, acc_g0 ? (g0 + Float4::load(grad_ptr0 + n * 4)) : g0);
        Float4::save(grad_ptr1 + n * 4, acc_g1 ? (g1 + Float4::load(grad_ptr1 + n * 4)) : g1);
    }
}

Status ArmBinaryCrossEntropyLayerGrad::OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                              LayerResource *resource, LayerParam *param, Context *context,
                                              LayerGradInfo *grad_info) {
    CHECK_PARAM_NULL(grad_info);
    if (grad_info->accumulate_blob_grad.size() < 2) {
        LOGD("ArmBinaryCrossEntropyLayerGrad::OnGrad, accumulate_blob_grad error\n");
        return Status(TNNERR_LAYER_ERR, "accumulate_blob_grad size error");
    }
    bool accumulate_blob_grad0 = grad_info->accumulate_blob_grad[0];
    bool accumulate_blob_grad1 = grad_info->accumulate_blob_grad[0];
    if (grad_info->upstream_grads.size() < 1) {
        LOGD("ArmBinaryCrossEntropyLayerGrad::OnGrad, upstream_grads error\n");
        return Status(TNNERR_LAYER_ERR, "upstream_grads size error");
    }
    Blob *upstream_grad = grad_info->upstream_grads[0];
    CHECK_PARAM_NULL(upstream_grad);

    if (inputs.size() != 3 || outputs.size() != 2) {
        return Status(TNNERR_LAYER_ERR, "input size or output size not match in ArmBinaryCrossEntropyLayerGrad");
    }

    auto fw_input0 = inputs[0];
    auto fw_input1 = inputs[1];
    auto fw_output = inputs[2];

    auto input0_dims = fw_input0->GetBlobDesc().dims;
    auto input1_dims = fw_input1->GetBlobDesc().dims;
    auto output_dims = fw_output->GetBlobDesc().dims;
    if (!DimsVectorUtils::Equal(input0_dims, input1_dims) || !DimsVectorUtils::Equal(input0_dims, output_dims)) {
        return Status(TNNERR_LAYER_ERR, "ArmBinaryCrossEntropyLayerGrad input dims and output dims not match");
    }

    auto grad0 = outputs[0];
    auto grad1 = outputs[1];
    auto dims0 = grad0->GetBlobDesc().dims;
    auto dims1 = grad1->GetBlobDesc().dims;

    if (!DimsVectorUtils::Equal(input0_dims, dims0) || !DimsVectorUtils::Equal(dims0, dims1)) {
        return Status(TNNERR_LAYER_ERR, "ArmBinaryCrossEntropyLayerGrad input dims and grad dims not match");
    }

    int batch   = DimsFunctionUtils::GetDim(dims0, 0);
    int channel = DimsFunctionUtils::GetDim(dims0, 1);

    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        int count      = batch * ROUND_UP(channel, 4) * DimsVectorUtils::Count(dims0, 2);
        int count_quad = UP_DIV(count, 4);

        auto input_ptr0        = reinterpret_cast<float *>(GetBlobHandlePtr(fw_input0->GetHandle()));
        auto input_ptr1        = reinterpret_cast<float *>(GetBlobHandlePtr(fw_input1->GetHandle()));
        auto grad_ptr0         = reinterpret_cast<float *>(GetBlobHandlePtr(grad0->GetHandle()));
        auto grad_ptr1         = reinterpret_cast<float *>(GetBlobHandlePtr(grad1->GetHandle()));
        auto upstream_grad_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(upstream_grad->GetHandle()));

        if (accumulate_blob_grad0) {
            if (accumulate_blob_grad1) {
                ExecBCEGrad<1, 1>(count_quad, input_ptr0, input_ptr1, grad_ptr0, grad_ptr1, upstream_grad_ptr);
            } else {
                ExecBCEGrad<1, 0>(count_quad, input_ptr0, input_ptr1, grad_ptr0, grad_ptr1, upstream_grad_ptr);
            }
        } else {
            if (accumulate_blob_grad1) {
                ExecBCEGrad<0, 1>(count_quad, input_ptr0, input_ptr1, grad_ptr0, grad_ptr1, upstream_grad_ptr);
            } else {
                ExecBCEGrad<0, 0>(count_quad, input_ptr0, input_ptr1, grad_ptr0, grad_ptr1, upstream_grad_ptr);
            }
        }
    } else {
        LOGE("ArmBinaryCrossEntropyLayerGrad::OnGrad, dtype not supported\n");
        return Status(TNNERR_LAYER_ERR, "dtype not supported");
    }

    return TNN_OK;
}

REGISTER_ARM_LAYER_GRAD(BinaryCrossEntropy, LAYER_BINARY_CROSSENTROPY)
REGISTER_ARM_GRAD_LAYOUT(LAYER_BINARY_CROSSENTROPY, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
