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
    const int axis        = param->axis;
    auto input_dims       = input->GetBlobDesc().dims;
    auto round_input_dims = GetNCXHWXRoundDims(input_dims, 4);
    const int batch       = DimsVectorUtils::Count(round_input_dims, 0, axis);
    const int slice_size  = DimsVectorUtils::Count(round_input_dims, axis + 1);
    const int slice_input = input_dims[axis];
    auto input_data       = reinterpret_cast<float *>(GetBlobHandlePtr(input->GetHandle()));

    for (int b = 0; b < batch; b++) {
        int slice_input_offset = 0;
        for (int i = 0; i < outputs.size(); i++) {
            auto output_blob = outputs[i];
            auto output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));
            const int slice  = output_blob->GetBlobDesc().dims[axis];

            auto output_data_ptr = output_data + b * slice * slice_size;
            auto input_data_ptr  = input_data + b * slice_input * slice_size + slice_input_offset * slice_size;

            memcpy(output_data_ptr, input_data_ptr, slice * slice_size * sizeof(float));
            slice_input_offset += slice;
        }
    }
    return 0;
}

static int splitv_channel(Blob *input, const std::vector<Blob *> &outputs, SplitVLayerParam *param) {
    const int axis              = param->axis;
    auto input_dims             = input->GetBlobDesc().dims;
    auto input_data             = reinterpret_cast<float *>(GetBlobHandlePtr(input->GetHandle()));
    DimsVector round_input_dims = GetNCXHWXRoundDims(input_dims, 4);

    int slice_offset = 0;
    for (int i = 0; i < outputs.size(); i++) {
        auto output                  = outputs[i];
        auto output_dims             = output->GetBlobDesc().dims;
        DimsVector round_output_dims = GetNCXHWXRoundDims(output_dims, 4);
        auto output_data             = reinterpret_cast<float *>(GetBlobHandlePtr(output->GetHandle()));
        const int slice              = output_dims[axis];
        auto plane                   = DimsVectorUtils::Count(output_dims, 2);
        for (int b = 0; b < output_dims[0]; b++) {
            auto input_b  = input_data + b * DimsVectorUtils::Count(round_input_dims, 1);
            auto output_b = output_data + b * DimsVectorUtils::Count(round_output_dims, 1);
            for (int c = 0; c < UP_DIV(output_dims[1], 4); c++) {
                auto output_z    = output_b + c * DimsVectorUtils::Count(round_output_dims, 2);
                auto input_c_idx = c * 4 + slice_offset;
                auto c_remain    = output_dims[1] - c * 4;
                auto c_c         = c_remain >= 4 ? 4 : c_remain;
                // both src and dst can use float4
                if (slice_offset % 4 == 0 && c * 4 + 3 < output_dims[1]) {
                    auto input_z = input_b + input_c_idx * plane;
                    for (int p = 0; p < plane; p++) {
                        Float4::save(output_z + p * 4, Float4::load(input_z + p * 4));
                    }
                } else {
                    int s;
                    for (s = 0; s < c_c; s++) {
                        auto src_start = ((input_c_idx + s) / 4) * plane * 4 + ((input_c_idx + s) % 4);
                        auto dst_start = s;
                        for (int p = 0; p < plane; p++)
                            output_z[dst_start + p * 4] = input_b[src_start + p * 4];
                    }
                    for (; s < 4; s++) {
                        for (int p = 0; p < plane; p++)
                            output_z[s + p * 4] = 0.f;
                    }
                }
            }
        }

        slice_offset += slice;
    }
    return 0;
}

static int splitv_channel_c4(Blob *input, const std::vector<Blob *> &outputs, SplitVLayerParam *param) {
    const int axis        = param->axis;
    auto input_dims       = input->GetBlobDesc().dims;
    auto round_input_dims = GetNCXHWXRoundDims(input_dims, 4);
    const int batch       = DimsVectorUtils::Count(round_input_dims, 0, axis);
    const int slice_size  = DimsVectorUtils::Count(round_input_dims, axis + 1);
    // different from split common, treat 4 element in channel as one
    const int slice_input = UP_DIV(input_dims[axis], 4);
    auto input_data       = reinterpret_cast<float *>(GetBlobHandlePtr(input->GetHandle()));

    for (int b = 0; b < batch; b++) {
        int slice_input_offset = 0;
        for (int i = 0; i < outputs.size(); i++) {
            auto output_blob = outputs[i];
            auto output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));
            // different from split common, treat 4 element in channel as one
            const int slice  = UP_DIV(output_blob->GetBlobDesc().dims[axis], 4);

            auto output_data_ptr = output_data + b * slice * slice_size;
            auto input_data_ptr  = input_data + b * slice_input * slice_size + slice_input_offset * slice_size;

            memcpy(output_data_ptr, input_data_ptr, slice * slice_size * sizeof(float));
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
                splitv_channel_c4(input_blob, outputs, layer_param);
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

}  // namespace TNN_NS
