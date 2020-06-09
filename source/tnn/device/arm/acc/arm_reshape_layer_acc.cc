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
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Reshape, LAYER_RESHAPE);

Status ArmReshapeLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs.size() < 1) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "layer's inputs size must >= 2");
    }

    auto in_data_type = inputs[0]->GetBlobDesc().data_type;

    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_input    = input->GetBlobDesc().dims;
    auto dims_output   = output->GetBlobDesc().dims;
    int data_byte_size = DataTypeUtils::GetBytesSize(output->GetBlobDesc().data_type);
    auto size_in_bytes = dims_input[0] * ROUND_UP(dims_input[1], 4) * dims_input[2] * dims_input[3] * data_byte_size;

    void *workspace = context_->GetSharedWorkSpace(size_in_bytes);
    char *input_origin  = GetBlobHandlePtr(input->GetHandle());
    char *output_origin = GetBlobHandlePtr(output->GetHandle());

    if (DATA_FORMAT_NC4HW4 == input->GetBlobDesc().data_format) {
        for (int b = 0; b < dims_output[0]; b++) {
            if (DATA_TYPE_FLOAT == in_data_type) {
                auto input_data =
                    reinterpret_cast<float *>(input_origin) + b * ROUND_UP(dims_input[1], 4) * dims_input[2] * dims_input[3];
                auto output_data = reinterpret_cast<float *>(output_origin) +
                                   b * ROUND_UP(dims_output[1], 4) * dims_output[2] * dims_output[3];
                UnpackC4(reinterpret_cast<float *>(workspace), input_data, dims_input[2] * dims_input[3], dims_input[1]);
                PackC4(output_data, reinterpret_cast<float *>(workspace), dims_output[2] * dims_output[3], dims_output[1]);
            } else if (DATA_TYPE_BFP16 == in_data_type) {
                auto input_data = reinterpret_cast<bfp16_t *>(input_origin) +
                                  b * ROUND_UP(dims_input[1], 4) * dims_input[2] * dims_input[3];
                auto output_data = reinterpret_cast<bfp16_t *>(output_origin) +
                                   b * ROUND_UP(dims_output[1], 4) * dims_output[2] * dims_output[3];
                UnpackC4(reinterpret_cast<bfp16_t *>(workspace), input_data, dims_input[2] * dims_input[3], dims_input[1]);
                PackC4(output_data, reinterpret_cast<bfp16_t *>(workspace), dims_output[2] * dims_output[3], dims_output[1]);
            } else {
                return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR int8 reshape, in todo list");
            }
        }
    } else {
        return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR nhwc/int8 fc, in todo list");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Reshape, LAYER_RESHAPE);

}  // namespace TNN_NS
