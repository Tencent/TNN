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
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(GridSample, LAYER_GRIDSAMPLE);

static inline bool within_bounds_2d(int32_t h, int32_t w, int32_t H, int32_t W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

static void ComputeNCHW(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_blob          = inputs[0];
    auto grid_blob           = inputs[1];
    auto output_blob         = outputs[0];
    auto input_dims          = input_blob->GetBlobDesc().dims;
    auto grid_dims           = grid_blob->GetBlobDesc().dims;
    auto output_dims         = output_blob->GetBlobDesc().dims;
    auto batch               = input_dims[0];
    auto channel             = input_dims[1];
    auto input_height        = input_dims[2];
    auto input_width         = input_dims[3];
    auto input_channel_area  = DimsVectorUtils::Count(input_dims, 2);
    auto grid_area           = DimsVectorUtils::Count(grid_dims, 1);
    auto output_channel_area = DimsVectorUtils::Count(output_dims, 2);
    auto input_base_ptr      = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));
    auto grid_base_ptr       = reinterpret_cast<float *>(GetBlobHandlePtr(grid_blob->GetHandle()));
    auto output_base_ptr     = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));

    for (int n = 0; n < batch; n++) {
        auto input_data_b  = input_base_ptr + n * channel * input_channel_area;
        auto grid_data_b   = grid_base_ptr + n * grid_area;
        auto output_data_b = output_base_ptr + n * channel * output_channel_area;
        OMP_PARALLEL_FOR_
        for (int i = 0; i < output_channel_area; ++i) {
            auto grid_position = grid_data_b + i * 2;
            float x            = grid_position[0];
            float y            = grid_position[1];
            // unnormalize
            float ix = (x + 1) * input_width * 0.5 - 0.5;
            float iy = (y + 1) * input_height * 0.5 - 0.5;
            // get corner pixel values from (x, y)
            // for 4d, we use north-east-south-west
            int ix_nw = static_cast<int>(std::floor(ix));
            int iy_nw = static_cast<int>(std::floor(iy));

            int ix_ne = ix_nw + 1;
            int iy_ne = iy_nw;

            int ix_sw = ix_nw;
            int iy_sw = iy_nw + 1;

            int ix_se = ix_nw + 1;
            int iy_se = iy_nw + 1;

            // get surfaces to each neighbor:
            float nw = (ix_se - ix) * (iy_se - iy);
            float ne = (ix - ix_sw) * (iy_sw - iy);
            float sw = (ix_ne - ix) * (iy - iy_ne);
            float se = (ix - ix_nw) * (iy - iy_nw);

            // calculate bilinear weighted pixel value and set output pixel
            float *input_data  = input_data_b;
            float *output_data = output_data_b + i;
            for (int c = 0; c < channel; ++c, output_data += output_channel_area, input_data += input_channel_area) {
                auto res = static_cast<float>(0);
                if (within_bounds_2d(iy_nw, ix_nw, input_height, input_width)) {
                    res += input_data[iy_nw * input_width + ix_nw] * nw;
                }
                if (within_bounds_2d(iy_ne, ix_ne, input_height, input_width)) {
                    res += input_data[iy_ne * input_width + ix_ne] * ne;
                }
                if (within_bounds_2d(iy_sw, ix_sw, input_height, input_width)) {
                    res += input_data[iy_sw * input_width + ix_sw] * sw;
                }
                if (within_bounds_2d(iy_se, ix_se, input_height, input_width)) {
                    res += input_data[iy_se * input_width + ix_se] * se;
                }
                *output_data = res;
            }
        }
    }
}

static void ComputeNC4HW4(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_blob          = inputs[0];
    auto grid_blob           = inputs[1];
    auto output_blob         = outputs[0];
    auto input_dims          = input_blob->GetBlobDesc().dims;
    auto grid_dims           = grid_blob->GetBlobDesc().dims;
    auto output_dims         = output_blob->GetBlobDesc().dims;
    auto batch               = input_dims[0];
    auto channel             = input_dims[1];
    auto input_height        = input_dims[2];
    auto input_width         = input_dims[3];
    auto input_channel_area  = DimsVectorUtils::Count(input_dims, 2);
    auto grid_area           = DimsVectorUtils::Count(grid_dims, 1);
    auto grid_hw             = DimsVectorUtils::Count(grid_dims, 2);
    auto output_channel_area = DimsVectorUtils::Count(output_dims, 2);
    auto channel_ud4         = UP_DIV(channel, 4);
    auto grid_channel_ud4    = UP_DIV(grid_dims[1], 4);
    auto input_base_ptr      = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));
    auto grid_base_ptr       = reinterpret_cast<float *>(GetBlobHandlePtr(grid_blob->GetHandle()));
    auto output_base_ptr     = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));
    bool grid_packed         = grid_blob->GetBlobDesc().data_format == DATA_FORMAT_NC4HW4;
    for (int n = 0; n < batch; n++) {
        auto input_data_b  = input_base_ptr + n * channel_ud4 * input_channel_area * 4;
        auto grid_data_b   = grid_base_ptr + n * grid_channel_ud4 * grid_hw * 4;
        auto output_data_b = output_base_ptr + n * channel_ud4 * output_channel_area * 4;
        RawBuffer reorder_grid_buffer;
        float *grid_buffer = nullptr;
        if (grid_packed) {
            reorder_grid_buffer = RawBuffer(grid_area * sizeof(float));
            UnpackC4(reorder_grid_buffer.force_to<float *>(), grid_data_b, DimsVectorUtils::Count(grid_dims, 2),
                     grid_dims[1]);
            grid_buffer = reorder_grid_buffer.force_to<float *>();
        }
        for (int c = 0; c < channel_ud4; c++) {
            OMP_PARALLEL_FOR_
            for (int i = 0; i < output_channel_area; ++i) {
                auto grid_position = grid_buffer + 2 * i;
                float x            = grid_position[0];
                float y            = grid_position[1];
                // unnormalize
                float ix = (x + 1) * input_width * 0.5 - 0.5;
                float iy = (y + 1) * input_height * 0.5 - 0.5;
                // get corner pixel values from (x, y)
                // for 4d, we use north-east-south-west
                int ix_nw = static_cast<int>(std::floor(ix));
                int iy_nw = static_cast<int>(std::floor(iy));

                int ix_ne = ix_nw + 1;
                int iy_ne = iy_nw;

                int ix_sw = ix_nw;
                int iy_sw = iy_nw + 1;

                int ix_se = ix_nw + 1;
                int iy_se = iy_nw + 1;

                // get surfaces to each neighbor:
                float nw = (ix_se - ix) * (iy_se - iy);
                float ne = (ix - ix_sw) * (iy_sw - iy);
                float sw = (ix_ne - ix) * (iy - iy_ne);
                float se = (ix - ix_nw) * (iy - iy_nw);

                Float4 nw_v = Float4(nw);
                Float4 ne_v = Float4(ne);
                Float4 sw_v = Float4(sw);
                Float4 se_v = Float4(se);

                // calculate bilinear weighted pixel value and set output pixel
                float *input_data  = input_data_b + c * input_channel_area * 4;
                float *output_data = output_data_b + c * output_channel_area * 4 + i * 4;
                Float4 res_v       = Float4(0.0f);
                if (within_bounds_2d(iy_nw, ix_nw, input_height, input_width)) {
                    res_v = res_v + Float4::load(input_data + iy_nw * input_width * 4 + ix_nw * 4) * nw_v;
                }
                if (within_bounds_2d(iy_ne, ix_ne, input_height, input_width)) {
                    res_v = res_v + Float4::load(input_data + iy_ne * input_width * 4 + ix_ne * 4) * ne_v;
                }
                if (within_bounds_2d(iy_sw, ix_sw, input_height, input_width)) {
                    res_v = res_v + Float4::load(input_data + iy_sw * input_width * 4 + ix_sw * 4) * sw_v;
                }
                if (within_bounds_2d(iy_se, ix_se, input_height, input_width)) {
                    res_v = res_v + Float4::load(input_data + iy_se * input_width * 4 + ix_se * 4) * se_v;
                }
                Float4::save(output_data, res_v);
            }
        }
    }
}

Status ArmGridSampleLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param      = dynamic_cast<GridSampleLayerParam *>(param_);
    auto input_blob = inputs[0];
    auto grid_blob  = inputs[1];
    auto input_dims = input_blob->GetBlobDesc().dims;
    auto grid_dims  = grid_blob->GetBlobDesc().dims;
    if (!(input_dims.size() == 4 && param->mode == 2 && param->pad_type == 0 && param->align_corners == 0)) {
        LOGE("Error: Arm layer acc don't support GridSample input size(%lu) or param:(%d, %d, %d)\n", input_dims.size(),
             param->mode, param->pad_type, param->align_corners);
        return Status(TNNERR_MODEL_ERR, "Error: Arm layer acc don't support.\n");
    }
    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto data_format = input_blob->GetBlobDesc().data_format;
        if (data_format == DATA_FORMAT_NC4HW4) {
            ComputeNC4HW4(inputs, outputs);
        } else if (data_format == DATA_FORMAT_NCHW) {
            ComputeNCHW(inputs, outputs);
        }
    } else {
        LOGE("Error: Arm layer acc don't support datatype: %d\n", inputs[0]->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: Arm layer acc don't support datatype\n");
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(GridSample, LAYER_GRIDSAMPLE);
REGISTER_ARM_LAYOUT(LAYER_GRIDSAMPLE, DATA_FORMAT_NCHW)
REGISTER_ARM_LAYOUT(LAYER_GRIDSAMPLE, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
