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

#include "tnn/device/arm/acc/Float4.h"
#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/acc/compute/pad_function.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Pad, LAYER_PAD);

Status ArmPadLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PadLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];
    auto input_dims  = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;
    PadUtils::PadContext pad_context;
    if (input_dims.size() != 4) {
        LOGE("Error: ArmPadLayerAcc only support 4 dims input, but now dims size is %lu/n", input_dims.size());
        return Status(TNNERR_MODEL_ERR, "Error: ArmPadLayerAcc only support 4 dims input");
    }
    pad_context.input_batch       = input_dims[0];
    pad_context.output_batch      = output_dims[0];
    pad_context.input_channel     = input_dims[1];
    pad_context.input_channel_r4  = ROUND_UP(input_dims[1], 4);
    pad_context.output_channel    = output_dims[1];
    pad_context.output_channel_r4 = ROUND_UP(output_dims[1], 4);
    pad_context.input_height      = input_dims[2];
    pad_context.output_height     = output_dims[2];
    pad_context.input_width       = input_dims[3];
    pad_context.output_width      = output_dims[3];
    const auto pads               = layer_param->pads;
    if (pads.size() != 6) {
        LOGE("Error: ArmPadLayerAcc layer acc does not support pas size %lu\n", pads.size());
        return Status(TNNERR_MODEL_ERR, "Error: ArmPadV2LayerAcc layer acc does not support");
    }
    pad_context.pad_l   = layer_param->pads[0];
    pad_context.pad_r   = layer_param->pads[1];
    pad_context.pad_t   = layer_param->pads[2];
    pad_context.pad_b   = layer_param->pads[3];
    pad_context.pad_c_b = layer_param->pads[4];
    pad_context.pad_c_e = layer_param->pads[5];
    pad_context.type    = layer_param->type;
    pad_context.value   = layer_param->value;

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT ||
        output_blob->GetBlobDesc().data_type == DATA_TYPE_INT32 ||
        output_blob->GetBlobDesc().data_type == DATA_TYPE_UINT32) {
        auto input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));
        auto output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));
        if (layer_param->type == 0) {
            // mode: const
            return PadUtils::ConstPadV2(input_data, output_data, input_dims, output_dims, pad_context);
        } else if (layer_param->type == 1) {
            // mode: reflect
            return PadUtils::ReflectPadV2(input_data, output_data, input_dims, output_dims, pad_context);
        } else {
            LOGE("Error: ArmPadLayerAcc does not support pad type:%d\n", layer_param->type);
            return Status(TNNERR_PARAM_ERR, "Error: ArmPadV2LayerAcc does not support pad type");
        }
    } else {
        LOGE("Error: ArmPadLayerAcc layer acc does not support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: ArmPadLayerAcc does not support datatype");
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(Pad, LAYER_PAD);
REGISTER_ARM_LAYOUT(LAYER_PAD, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
