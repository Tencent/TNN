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

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/x86_device.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/device/x86/acc/compute/x86_compute_int8.h"

namespace TNN_NS {

DECLARE_X86_ACC(Concat, LAYER_CONCAT);

Status X86ConcatLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ConcatLayerParam *>(param_);
    if (!param) {
        LOGE("Error: ConcatLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: ConcatLayerParam is nil");
    }
    if (inputs.size() < 2) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "Concat layer's inputs size must >= 2");
    }

    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        switch (param->axis) {
            case 1:
                X86ConcatChannelInt8(outputs[0], inputs);
                break;
            default:
                X86ConcatCommonInt8(outputs[0], inputs, param->axis);
                break;
        }
        return TNN_OK;
    }

    auto input  = inputs[0];
    auto output = outputs[0];
    auto dims   = input->GetBlobDesc().dims;

    const int axis = param->axis;
    if (axis > dims.size() || axis < 0) {
        LOGE("Error: Concat layer param invalid\n");
        return Status(TNNERR_PARAM_ERR, "Concat layer param invalid");
    }

    int num_concats = 1;
    for (int i = 0; i < axis; i++) {
        num_concats *= dims[i];
    }

    int concate_size = 1;
    for (int i = axis + 1; i < dims.size(); i++) {
        concate_size *= dims[i];
    }

    auto datasize                 = DataTypeUtils::GetBytesSize(input->GetBlobDesc().data_type);
    int8_t *output_data           = handle_ptr<int8_t *>(output->GetHandle());
    int output_concat_axis        = output->GetBlobDesc().dims[axis];
    int output_concat_axis_offset = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        // use int8_t for all types
        int8_t *input_data          = handle_ptr<int8_t *>(inputs[i]->GetHandle());
        const int input_concat_axis = inputs[i]->GetBlobDesc().dims[axis];
        for (int n = 0; n < num_concats; ++n) {
            memcpy(output_data + (n * output_concat_axis + output_concat_axis_offset) * concate_size * datasize,
                   input_data + n * input_concat_axis * concate_size * datasize,
                   input_concat_axis * concate_size * datasize);
        }
        output_concat_axis_offset += input_concat_axis;
    }
    return TNN_OK;
}

REGISTER_X86_ACC(Concat, LAYER_CONCAT);

}