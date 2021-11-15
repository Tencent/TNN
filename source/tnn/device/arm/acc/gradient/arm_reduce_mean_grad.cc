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

DECLARE_ARM_LAYER_GRAD(ReduceMean, LAYER_REDUCE_MEAN);

Status ArmReduceMeanLayerGrad::OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                      LayerResource *resource, LayerParam *param, Context *context,
                                      LayerGradInfo *grad_info) {
    CHECK_PARAM_NULL(grad_info);
    if (grad_info->accumulate_blob_grad.size() < 1) {
        LOGD("ArmReduceMeanLayerGrad::OnGrad, accumulate_blob_grad error\n");
        return Status(TNNERR_LAYER_ERR, "accumulate_blob_grad size error");
    }
    bool accumulate_blob_grad = grad_info->accumulate_blob_grad[0];
    if (grad_info->upstream_grads.size() < 1) {
        LOGD("ArmReduceMeanLayerGrad::OnGrad, upstream_grads error\n");
        return Status(TNNERR_LAYER_ERR, "upstream_grads size error");
    }
    Blob *upstream_grad = grad_info->upstream_grads[0];

    auto fw_input  = inputs[0];
    auto fw_output = inputs[1];

    int input_count  = DimsVectorUtils::Count(fw_input->GetBlobDesc().dims);
    int output_count = DimsVectorUtils::Count(fw_output->GetBlobDesc().dims);

    if (output_count != 1) {
        LOGE("ArmReduceMeanLayerGrad::OnGrad, only all reduce supported yet, output count is %d\n", output_count);
        return Status(TNNERR_LAYER_ERR, "only all reduce supported yet");
    }

    auto output = outputs[0];
    auto dims   = output->GetBlobDesc().dims;

    int batch      = DimsFunctionUtils::GetDim(dims, 0);
    int channel    = DimsFunctionUtils::GetDim(dims, 1);
    int count      = batch * ROUND_UP(channel, 4) * DimsVectorUtils::Count(dims, 2);
    int count_quad = UP_DIV(count, 4);

    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        Float4 grad = Float4(float(output_count) / float(input_count));

        if (upstream_grad) {
            float *ptr = (float *)GetBlobHandlePtr(upstream_grad->GetHandle());
            grad       = grad * ptr[0];
        }

        auto output_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(output->GetHandle()));

        if (!accumulate_blob_grad) {
            OMP_PARALLEL_FOR_
            for (int n = 0; n < count_quad; n++) {
                Float4::save(output_ptr + n * 4, grad);
            }
        } else {
            OMP_PARALLEL_FOR_
            for (int n = 0; n < count_quad; n++) {
                Float4::save(output_ptr + n * 4, grad + Float4::load(output_ptr + n * 4));
            }
        }
    } else {
        LOGE("ArmReduceMeanLayerGrad::OnGrad, dtype not supported\n");
        return Status(TNNERR_LAYER_ERR, "dtype not supported");
    }

    return TNN_OK;
}

REGISTER_ARM_LAYER_GRAD(ReduceMean, LAYER_REDUCE_MEAN)
REGISTER_ARM_GRAD_LAYOUT(LAYER_REDUCE_MEAN, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
