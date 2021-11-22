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

#include "tnn/device/arm/acc/compute/blob_compute.h"
#include "tnn/device/arm/acc/gradient/arm_gradient_layer_acc.h"

namespace TNN_NS {

DECLARE_ARM_LAYER_GRAD(Concat, LAYER_CONCAT);

Status ArmConcatLayerGrad::OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                  LayerResource *resource, LayerParam *param, Context *context,
                                  const LayerGradInfo &grad_info) {
    int inputs_count = inputs.size() - 2;
    ON_GRAD_PREPARATION_IOR(inputs_count, 1, 0);

    auto concat_param = dynamic_cast<ConcatLayerParam *>(param);
    CHECK_PARAM_NULL(concat_param);
    const int axis = concat_param->axis;

    bool is_chanel_c4 = false;
    if (axis == 1) {
        is_chanel_c4 = true;
        for (int i = 0; i < fw_inputs.size() - 1; i++) {
            if (input_dims[i][1] % 4) {
                is_chanel_c4 = false;
                break;
            }
        }
    }

    if (fw_inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        if (axis == 1) {
            if (is_chanel_c4) {
                SplitvChannelC4(output_grads[0], input_grads, axis);
            } else {
                SplitvChannel(output_grads[0], input_grads, axis);
            }
        } else {
            SplitvCommon(output_grads[0], input_grads, axis);
        }
    } else {
        LOGE("ArmConcatLayerGrad::OnGrad, dtype not supported\n");
        return Status(TNNERR_TRAIN_ERROR, "dtype not supported");
    }

    return TNN_OK;
}

REGISTER_ARM_LAYER_GRAD(Concat, LAYER_CONCAT);
REGISTER_ARM_GRAD_LAYOUT(LAYER_CONCAT, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
