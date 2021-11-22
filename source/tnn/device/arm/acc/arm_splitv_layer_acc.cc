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
#include "tnn/device/arm/acc/compute/blob_compute.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(SplitV, LAYER_SPLITV);

Status ArmSplitVLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<SplitVLayerParam *>(param_);
    if (!layer_param || layer_param->slices.size() != outputs.size()) {
        return Status(TNNERR_PARAM_ERR, "ArmSplitVLayerAcc has invalid param, slices size != output blobs size");
    }

    const int axis    = layer_param->axis;
    auto input_blob   = inputs[0];
    bool is_chanel_c4 = false;
    if (axis == 1) {
        is_chanel_c4 = true;
        for (int i = 0; i < outputs.size() - 1; i++) {
            auto output_dims = outputs[i]->GetBlobDesc().dims;
            if (output_dims[1] % 4) {
                is_chanel_c4 = false;
                break;
            }
        }
    }

    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        if (axis == 1) {
            if (is_chanel_c4) {
                SplitvChannelC4(input_blob, outputs, axis);
            } else {
                SplitvChannel(input_blob, outputs, axis);
            }
        } else {
            SplitvCommon(input_blob, outputs, axis);
        }
    } else if (input_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        LOGE("Error: layer acc don't support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(SplitV, LAYER_SPLITV);
REGISTER_ARM_LAYOUT(LAYER_SPLITV, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
