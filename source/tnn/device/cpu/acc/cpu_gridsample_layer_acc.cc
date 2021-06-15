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

#include <cmath>

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(GridSample, LAYER_GRIDSAMPLE);

Status CpuGridSampleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

static inline bool within_bounds_2d(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

Status CpuGridSampleLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<GridSampleLayerParam *>(param_);
    if (layer_param->mode != 2 || layer_param->pad_type != 0 || layer_param->align_corners != 0) {
        return Status(TNNERR_PARAM_ERR, "CpuGridSampleLayerAcc dont support some mode or pade type or align_corners");
    }
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto grid_dims   = inputs[1]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    if (output_dims.size() != 4) {
        return Status(TNNERR_PARAM_ERR, "CpuGridSampleLayerAcc only support 4D sampler");
    }
    const int batch               = input_dims[0];
    const int channel             = input_dims[1];
    const int input_height        = input_dims[2];
    const int input_width         = input_dims[3];
    const int input_channel_area  = DimsVectorUtils::Count(input_dims, 2);
    const int output_channel_area = DimsVectorUtils::Count(output_dims, 2);
    const int grid_height         = grid_dims[1];
    const int grid_width          = grid_dims[2];
    const int grid_area           = DimsVectorUtils::Count(grid_dims, 1);
    auto output_height            = output_dims[2];
    auto output_width             = output_dims[3];

    float *input_base_ptr  = (float *)((char *)inputs[0]->GetHandle().base + inputs[0]->GetHandle().bytes_offset);
    float *grid_base_ptr   = (float *)((char *)inputs[1]->GetHandle().base + inputs[1]->GetHandle().bytes_offset);
    float *output_base_ptr = (float *)((char *)outputs[0]->GetHandle().base + outputs[0]->GetHandle().bytes_offset);

    if (inputs[0]->GetBlobDesc().data_type != DATA_TYPE_FLOAT) {
        return Status(TNNERR_PARAM_ERR, "CpuGridSampleLayerAcc now only support float data");
    }
    for (int n = 0; n < batch; n++) {
        auto input_data_b  = input_base_ptr + n * channel * input_channel_area;
        auto grid_data_b   = grid_base_ptr + n * grid_area;
        auto output_data_b = output_base_ptr + n * channel * output_channel_area;
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                auto grid_position = grid_data_b + h * grid_width * 2 + w * 2;
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
                bool nw_within_bound = within_bounds_2d(iy_nw, ix_nw, input_height, input_width);
                bool ne_within_bound = within_bounds_2d(iy_ne, ix_ne, input_height, input_width);
                bool sw_within_bound = within_bounds_2d(iy_sw, ix_sw, input_height, input_width);
                bool se_within_bound = within_bounds_2d(iy_se, ix_se, input_height, input_width);
                float nw             = nw_within_bound ? (ix_se - ix) * (iy_se - iy) : 0;
                float ne             = ne_within_bound ? (ix - ix_sw) * (iy_sw - iy) : 0;
                float sw             = sw_within_bound ? (ix_ne - ix) * (iy - iy_ne) : 0;
                float se             = se_within_bound ? (ix - ix_nw) * (iy - iy_nw) : 0;
                int nw_index         = nw_within_bound ? iy_nw * input_width + ix_nw : 0;
                int ne_index         = ne_within_bound ? iy_ne * input_width + ix_ne : 0;
                int sw_index         = sw_within_bound ? iy_sw * input_width + ix_sw : 0;
                int se_index         = se_within_bound ? iy_se * input_width + ix_se : 0;

                // calculate bilinear weighted pixel value and set output pixel
                float *input_data  = input_data_b;
                float *output_data = output_data_b + h * output_width + w;
                for (int c = 0; c < channel;
                     ++c, output_data += output_channel_area, input_data += input_channel_area) {
                    auto res = static_cast<float>(0);
                    res += input_data[nw_index] * nw;
                    res += input_data[ne_index] * ne;
                    res += input_data[sw_index] * sw;
                    res += input_data[se_index] * se;
                    *output_data = res;
                }
            }
        }
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(GridSample, LAYER_GRIDSAMPLE);

}  // namespace TNN_NS
