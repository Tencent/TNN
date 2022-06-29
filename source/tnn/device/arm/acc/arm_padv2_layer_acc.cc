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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/pad_utils.h"

namespace TNN_NS {
using namespace arm;

DECLARE_ARM_ACC(PadV2, LAYER_PADV2);

Status ArmPadV2LayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PadLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: layer param is nil");
    }
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];
    auto input_dims  = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;
    PadUtils::PadContext pad_context;
    if (input_dims.size() < 2 || input_dims.size() > 5) {
        LOGE("Error: ArmPadV2LayerAcc layer acc does not support input dims size %lu\n", input_dims.size());
        return Status(TNNERR_MODEL_ERR, "Error: ArmPadV2LayerAcc layer acc does not support;");
    }
    pad_context.input_batch  = input_dims[0];
    pad_context.output_batch = output_dims[0];

    if (input_dims.size() >= 2) {
        pad_context.input_channel     = input_dims[1];
        pad_context.input_channel_r4  = ROUND_UP(input_dims[1], 4);
        pad_context.output_channel    = output_dims[1];
        pad_context.output_channel_r4 = ROUND_UP(output_dims[1], 4);
    }
    if (input_dims.size() >= 3) {
        pad_context.input_height  = input_dims[2];
        pad_context.output_height = output_dims[2];
    }
    if (input_dims.size() >= 4) {
        pad_context.input_width  = input_dims[3];
        pad_context.output_width = output_dims[3];
    }
    if (input_dims.size() == 5) {
        pad_context.input_depth  = input_dims[2];
        pad_context.input_height = input_dims[3];
        pad_context.input_width  = input_dims[4];
        pad_context.output_depth = output_dims[2];
        pad_context.input_height = output_dims[3];
        pad_context.output_width = output_dims[4];
    }
    const auto pads = layer_param->pads;
    if (pads.size() < 2 || pads.size() > 10) {
        LOGE("Error: ArmPadV2LayerAcc layer acc does not support pas size %lu\n", pads.size());
        return Status(TNNERR_MODEL_ERR, "Error: ArmPadV2LayerAcc layer acc does not support");
    }
    switch (pads.size()) {
        case 4: {
            pad_context.pad_b_b = pads[0];  // pad batch begin
            pad_context.pad_c_b = pads[1];  // pad channel begin
            pad_context.pad_b_e = pads[2];  // pad batch end
            pad_context.pad_c_e = pads[3];  // pad channel end
            break;
        }
        case 6: {
            pad_context.pad_b_b = pads[0];  // pad batch begin
            pad_context.pad_c_b = pads[1];  // pad channel begin
            pad_context.pad_t   = pads[2];  // pad height begin
            pad_context.pad_b_e = pads[3];  // pad batch end
            pad_context.pad_c_e = pads[4];  // pad channel end
            pad_context.pad_b   = pads[5];  // pad height end
            break;
        }
        case 8: {
            pad_context.pad_b_b = pads[0];  // pad batch begin
            pad_context.pad_c_b = pads[1];  // pad channel begin
            pad_context.pad_t   = pads[2];  // pad height begin
            pad_context.pad_l   = pads[3];  // pad width begin
            pad_context.pad_b_e = pads[4];  // pad batch end
            pad_context.pad_c_e = pads[5];  // pad channel end
            pad_context.pad_b   = pads[6];  // pad height end
            pad_context.pad_r   = pads[7];  // pad width end
            break;
        }
        case 10: {
            pad_context.pad_b_b = pads[0];  // pad batch begin
            pad_context.pad_c_b = pads[1];  // pad channel begin
            pad_context.pad_d_b = pads[2];  // pad depth begin
            pad_context.pad_t   = pads[3];  // pad height begin
            pad_context.pad_l   = pads[4];  // pad width begin
            pad_context.pad_b_e = pads[5];  // pad batch end
            pad_context.pad_c_e = pads[6];  // pad channel end
            pad_context.pad_d_e = pads[7];  // pad depth end
            pad_context.pad_b   = pads[8];  // pad height end
            pad_context.pad_r   = pads[9];  // pad width end
            break;
        }
    }

    pad_context.type  = layer_param->type;
    pad_context.value = layer_param->value;
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
            LOGE("Error: ArmPadV2LayerAcc does not support pad type: type:%d\n", layer_param->type);
            return Status(TNNERR_PARAM_ERR, "Error: ArmPadV2LayerAcc layer param does not support pad type");
        }
    } else {
        LOGE("Error: ArmPadV2LayerAcc does not support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: ArmPadV2LayerAcc does not support datatype");
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(PadV2, LAYER_PADV2);
REGISTER_ARM_LAYOUT(LAYER_PADV2, DATA_FORMAT_NC4HW4);

}  // namespace TNN_NS