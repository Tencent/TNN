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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(SplitV, LAYER_SPLITV);

static DimsVector GetNCXHWXRoundDims(const DimsVector &dims, const int round) {
    DimsVector round_dims = {dims[0], UP_DIV(dims[1], round)};
    for (int i = 2; i < dims.size(); ++i) {
        round_dims.push_back(dims[i]);
    }
    round_dims.push_back(round);
    return round_dims;
}

// batch || height || width, no channel
static int splitv_common(Blob *input, const std::vector<Blob *> &outputs, SplitVLayerParam *param) {
    const auto &data_type = input->GetBlobDesc().data_type;
    const int axis        = param->axis;
    auto input_dims       = input->GetBlobDesc().dims;
    int round_size        = 0;
    int byte_size         = 0;
    if (data_type == DATA_TYPE_FLOAT) {
        round_size = 4;
        byte_size  = sizeof(float);
    } else {
        round_size = 8;
        byte_size  = sizeof(fp16_t);
    }
    auto round_input_dims = GetNCXHWXRoundDims(input_dims, round_size);

    const int batch       = DimsVectorUtils::Count(round_input_dims, 0, axis);
    const int slice_size  = DimsVectorUtils::Count(round_input_dims, axis + 1);
    const int slice_input = input_dims[axis];
    char *input_data      = GetBlobHandlePtr(input->GetHandle());

    for (int b = 0; b < batch; b++) {
        int slice_input_offset = 0;
        for (int i = 0; i < outputs.size(); i++) {
            auto output_blob      = outputs[i];
            char *output_data     = GetBlobHandlePtr(output_blob->GetHandle());
            const int slice       = output_blob->GetBlobDesc().dims[axis];
            char *output_data_ptr = output_data + b * slice * slice_size * byte_size;
            char *input_data_ptr  = input_data + (b * slice_input + slice_input_offset) * slice_size * byte_size;

            memcpy(output_data_ptr, input_data_ptr, slice * slice_size * byte_size);
            slice_input_offset += slice;
        }
    }
    return 0;
}

static int splitv_channel(Blob *input, const std::vector<Blob *> &outputs, SplitVLayerParam *param) {
    const int axis   = param->axis;
    auto input_dims  = input->GetBlobDesc().dims;
    char *input_data = GetBlobHandlePtr(input->GetHandle());
    auto data_type   = input->GetBlobDesc().data_type;
    int round_size   = 0;
    int byte_size    = 0;
    if (data_type == DATA_TYPE_FLOAT) {
        round_size = 4;
        byte_size  = sizeof(float);
    } else {
        round_size = 8;
        byte_size  = sizeof(fp16_t);
    }
    DimsVector round_input_dims = GetNCXHWXRoundDims(input_dims, round_size);

    int slice_offset = 0;
    for (int i = 0; i < outputs.size(); i++) {
        auto output                  = outputs[i];
        auto output_dims             = output->GetBlobDesc().dims;
        DimsVector round_output_dims = GetNCXHWXRoundDims(output_dims, round_size);
        char *output_data            = GetBlobHandlePtr(output->GetHandle());
        const int slice              = output_dims[axis];
        auto plane                   = DimsVectorUtils::Count(output_dims, 2);
        for (int b = 0; b < output_dims[0]; b++) {
            char *input_b  = input_data + b * DimsVectorUtils::Count(round_input_dims, 1) * byte_size;
            char *output_b = output_data + b * DimsVectorUtils::Count(round_output_dims, 1) * byte_size;
            for (int c = 0; c < UP_DIV(output_dims[1], round_size); c++) {
                char *output_z   = output_b + c * DimsVectorUtils::Count(round_output_dims, 2) * byte_size;
                auto input_c_idx = c * round_size + slice_offset;
                auto c_remain    = output_dims[1] - c * round_size;
                auto c_c         = c_remain >= round_size ? round_size : c_remain;
                // both src and dst can use float4
                if (slice_offset % round_size == 0 && (c + 1) * round_size <= output_dims[1]) {
                    char *input_z = input_b + input_c_idx * plane * byte_size;
                    for (int p = 0; p < plane; p++) {
                        memcpy(output_z + p * round_size * byte_size, input_z + p * round_size * byte_size,
                               round_size * byte_size);
                    }
                } else {
                    int s = 0;
                    for (; s < c_c; s++) {
                        auto src_start =
                            ((input_c_idx + s) / round_size) * plane * round_size + ((input_c_idx + s) % round_size);
                        auto dst_start = s;
                        for (int p = 0; p < plane; p++) {
                            memcpy(output_z + (dst_start + p * round_size) * byte_size,
                                   input_b + (src_start + p * round_size) * byte_size, byte_size);
                        }
                    }
                    for (; s < round_size; s++) {
                        for (int p = 0; p < plane; p++) {
                            memset(output_z + (s + p * round_size) * byte_size, 0.0f, byte_size);
                        }
                    }
                }
            }
        }

        slice_offset += slice;
    }
    return 0;
}

static int splitv_channel_round(Blob *input, const std::vector<Blob *> &outputs, SplitVLayerParam *param) {
    const int axis  = param->axis;
    auto input_dims = input->GetBlobDesc().dims;
    auto data_type  = input->GetBlobDesc().data_type;
    int round_size  = 0;
    int byte_size   = 0;
    if (data_type == DATA_TYPE_FLOAT) {
        round_size = 4;
        byte_size  = sizeof(float);
    } else {
        round_size = 8;
        byte_size  = sizeof(fp16_t);
    }
    auto round_input_dims = GetNCXHWXRoundDims(input_dims, round_size);
    const int batch       = DimsVectorUtils::Count(round_input_dims, 0, axis);
    const int slice_size  = DimsVectorUtils::Count(round_input_dims, axis + 1);
    // different from split common, treat 4 element in channel as one
    const int slice_input = UP_DIV(input_dims[axis], round_size);
    char *input_data      = GetBlobHandlePtr(input->GetHandle());

    for (int b = 0; b < batch; b++) {
        int slice_input_offset = 0;
        for (int i = 0; i < outputs.size(); i++) {
            auto output_blob  = outputs[i];
            char *output_data = GetBlobHandlePtr(output_blob->GetHandle());
            // different from split common, treat 4 element in channel as one
            const int slice       = UP_DIV(output_blob->GetBlobDesc().dims[axis], round_size);
            char *output_data_ptr = output_data + b * slice * slice_size * byte_size;
            char *input_data_ptr  = input_data + (b * slice_input + slice_input_offset) * slice_size * byte_size;
            memcpy(output_data_ptr, input_data_ptr, slice * slice_size * byte_size);
            slice_input_offset += slice;
        }
    }
    return 0;
}

Status ArmSplitVLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<SplitVLayerParam *>(param_);
    if (!layer_param || layer_param->slices.size() != outputs.size()) {
        return Status(TNNERR_PARAM_ERR, "ArmSplitVLayerAcc has invalid param, slices size != output blobs size");
    }
    const int axis         = layer_param->axis;
    const auto &input_blob = inputs[0];
    const auto &data_type  = input_blob->GetBlobDesc().data_type;
    bool is_channel_round  = false;
    int round_size         = 0;
    if (data_type == DATA_TYPE_FLOAT) {
        round_size = 4;
    } else {
        round_size = 8;
    }
    if (axis == 1) {
        is_channel_round = true;
        for (int i = 0; i < outputs.size() - 1; i++) {
            auto output_dims = outputs[i]->GetBlobDesc().dims;
            if (output_dims[1] % round_size) {
                is_channel_round = false;
                break;
            }
        }
    }

    if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_HALF) {
        if (axis == 1) {
            if (is_channel_round) {
                splitv_channel_round(input_blob, outputs, layer_param);
            } else {
                splitv_channel(input_blob, outputs, layer_param);
            }
        } else {
            splitv_common(input_blob, outputs, layer_param);
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
REGISTER_ARM_PRECISION_FP16(LAYER_SPLITV);

}  // namespace TNN_NS
