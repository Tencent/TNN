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

#include "tnn/device/arm/acc/arm_reshape_layer_acc.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

Status ArmReshapeLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs.size() < 1) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "layer's inputs size must >= 2");
    }

    auto in_data_type = inputs[0]->GetBlobDesc().data_type;

    auto input  = inputs[0];
    auto output = outputs[0];

    int data_byte_size = DataTypeUtils::GetBytesSize(output->GetBlobDesc().data_type);
    auto size_in_bytes = DimsVectorUtils::Count(input->GetBlobDesc().dims) * data_byte_size;
    workspace_         = context_->GetSharedWorkSpace(size_in_bytes);

    auto param    = (ReshapeLayerParam *)param_;
    ASSERT(param != nullptr);

    if (DATA_FORMAT_NC4HW4 == input->GetBlobDesc().data_format) {
        if (DATA_TYPE_FLOAT == in_data_type) {
            return Exec<float>(inputs, outputs);
        } else if (DATA_TYPE_BFP16 == in_data_type) {
            return Exec<bfp16_t>(inputs, outputs);
        } else {
            return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR int8 reshape, in todo list");
        }
    } else {
        return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR nhwc/int8 fc, in todo list");
    }
}

template <typename T>
Status ArmReshapeLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto dims_input   = inputs[0]->GetBlobDesc().dims;
    auto dims_output  = outputs[0]->GetBlobDesc().dims;

    char *input_origin  = GetBlobHandlePtr(inputs[0]->GetHandle());
    char *output_origin = GetBlobHandlePtr(outputs[0]->GetHandle());

    auto param    = (ReshapeLayerParam *)param_;
    ASSERT(param != nullptr);

    auto ic    = dims_input[1];
    auto ic_r4 = ROUND_UP(dims_input[1], 4);
    auto ih    = dims_input[2];
    auto iw    = dims_input[3];
    auto oc    = dims_output[1];
    auto oc_r4 = ROUND_UP(dims_output[1], 4);
    auto oh    = dims_output[2];
    auto ow    = dims_output[3];

    auto input_plane     = ic * ih * iw;
    auto input_plane_r4  = ic_r4 * ih * iw;
    auto output_plane    = oc * oh * ow;
    auto output_plane_r4 = oc_r4 * oh * ow;

    for (int b = 0; b < dims_input[0]; b++) {
        auto input_data =
            reinterpret_cast<float *>(input_origin) + b * input_plane_r4;
        auto workspace_data =
            reinterpret_cast<float *>(workspace_) + b * input_plane;
        if (param->reshape_type == 0)
            UnpackC4(workspace_data, input_data, ih * iw, ic);
        else if (param->reshape_type == 1)
            UnpackC4ToNHWC(workspace_data, input_data, ih * iw, ic);
        else
            return Status(TNNERR_LAYER_ERR, "Unsupport reshape type");
    }
    for (int b = 0; b < dims_output[0]; b++) {
        auto workspace_data =
            reinterpret_cast<float *>(workspace_) + b * output_plane;
        auto output_data =
            reinterpret_cast<float *>(output_origin) + b * output_plane_r4;
        if (param->reshape_type == 0)
            PackC4(output_data, workspace_data, oh * ow, oc);
        else if (param->reshape_type == 1)
            PackC4FromNHWC(output_data, workspace_data, oh * ow, oc);
        else
            return Status(TNNERR_LAYER_ERR, "Unsupport reshape type");
    }

    return TNN_OK;
};

REGISTER_ARM_ACC(Reshape, LAYER_RESHAPE);

}  // namespace TNN_NS
