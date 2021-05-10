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
    auto input_base_ptr      = reinterpret_cast<float *>(input_blob->GetHandle().base);
    auto grid_base_ptr       = reinterpret_cast<float *>(grid_blob->GetHandle().base);
    auto output_base_ptr     = reinterpret_cast<float *>(output_blob->GetHandle().base);
    for (int n = 0; n < batch; n++) {
        auto input_data  = input_base_ptr + n * channel * input_channel_area;
        auto grid_data   = grid_base_ptr + n * grid_area;
        auto output_data = output_base_ptr + n * channel * output_channel_area;
        OMP_PARALLEL_FOR_
        for (int i = 0; i < output_channel_area; ++i) {
            auto grid_position = grid_data + 2 * i;
            float grid_x       = (grid_position[0] + 1) * input_width * 0.5 - 0.5;
            float grid_y       = (grid_position[1] + 1) * input_height * 0.5 - 0.5;
            // w0lambda: (1-x) ; w1lambda: x
            const int w0   = std::floor(grid_x);
            const int w1p  = (w0 < input_width - 1) ? 1 : 0;
            float w1lambda = grid_x - w0;
            float w0lambda = (float)1. - w1lambda;
            if (w0 < 0 || w0 > input_width - 1) {
                w0lambda = 0;
            }
            if (w0 + 1 < 0 || w0 + 1 > input_width - 1) {
                w1lambda = 0;
            }
            // h0lambda: (1-y) ; h1lambda: y
            const int h0   = std::floor(grid_y);
            const int h1p  = (h0 < input_height - 1) ? 1 : 0;
            float h1lambda = grid_y - h0;
            float h0lambda = (float)1. - h1lambda;
            if (h0 < 0 || h0 > input_height - 1) {
                h0lambda = 0;
            }
            if (h0 + 1 < 0 || h0 + 1 > input_height - 1) {
                h1lambda = 0;
            }
            // Note: read outside valid roi will raise
            const float *x_data_ptr = input_data + h0 * input_width + w0;
            float *y_data_ptr       = output_data + i;
            for (int c = 0; c < channel; c++) {
                // reference: https://zh.wikipedia.org/wiki/%E5%8F%8C%E7%BA%BF%E6%80%A7%E6%8F%92%E5%80%BC
                // f(x,y) = (1-x) * (1-y) * f(0,0) + (1-x) * (y) * f(0,1) + (x) *(1-y) * f(1,0) + (x) * (y) * f(1,1)
                y_data_ptr[0] = h0lambda * w0lambda * x_data_ptr[0] + h0lambda * w1lambda * x_data_ptr[w1p] +
                                h1lambda * w0lambda * x_data_ptr[h1p * input_width] +
                                h1lambda * w1lambda * x_data_ptr[h1p * input_width + w1p];
                x_data_ptr += input_channel_area;
                y_data_ptr += output_channel_area;
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
    auto input_base_ptr      = reinterpret_cast<float *>(input_blob->GetHandle().base);
    auto grid_base_ptr       = reinterpret_cast<float *>(grid_blob->GetHandle().base);
    auto output_base_ptr     = reinterpret_cast<float *>(output_blob->GetHandle().base);
    bool grid_packed         = grid_blob->GetBlobDesc().data_format == DATA_FORMAT_NC4HW4;
    for (int n = 0; n < batch; n++) {
        auto input_data    = input_base_ptr + n * channel_ud4 * input_channel_area * 4;
        auto grid_data     = grid_base_ptr + n * grid_channel_ud4 * grid_hw * 4;
        auto output_data   = output_base_ptr + n * channel_ud4 * output_channel_area * 4;
        RawBuffer reorder_grid_buffer;
        float *grid_buffer = nullptr;
        if (grid_packed) {
            reorder_grid_buffer = RawBuffer(grid_area * sizeof(float));
            UnpackC4(reorder_grid_buffer.force_to<float *>(), grid_data, DimsVectorUtils::Count(grid_dims, 2),
                     grid_dims[1]);
            grid_buffer = reorder_grid_buffer.force_to<float *>();
        }
        for (int c = 0; c < channel_ud4; c++) {
            OMP_PARALLEL_FOR_
            for (int i = 0; i < output_channel_area; ++i) {
                auto grid_position = grid_buffer + 2 * i;
                float grid_x       = (grid_position[0] + 1) * input_width * 0.5 - 0.5;
                float grid_y       = (grid_position[1] + 1) * input_height * 0.5 - 0.5;
                // w0lambda: (1-x) ; w1lambda: x
                const int w0   = std::floor(grid_x);
                const int w1p  = (w0 < input_width - 1) ? 1 : 0;
                float w1lambda = grid_x - w0;
                float w0lambda = (float)1. - w1lambda;
                if (w0 < 0 || w0 > input_width - 1) {
                    w0lambda = 0;
                }
                if (w0 + 1 < 0 || w0 + 1 > input_width - 1) {
                    w1lambda = 0;
                }
                // h0lambda: (1-y) ; h1lambda: y
                const int h0   = std::floor(grid_y);
                const int h1p  = (h0 < input_height - 1) ? 1 : 0;
                float h1lambda = grid_y - h0;
                float h0lambda = (float)1. - h1lambda;
                if (h0 < 0 || h0 > input_height - 1) {
                    h0lambda = 0;
                }
                if (h0 + 1 < 0 || h0 + 1 > input_height - 1) {
                    h1lambda = 0;
                }
                // Note: read outside valid roi will raise
                auto x_data_ptr   = input_data + c * input_channel_area * 4 + h0 * input_width * 4 + w0 * 4;
                auto y_data_ptr   = output_data + c * output_channel_area * 4 + i * 4;
                Float4 w0lambda_v = Float4(w0lambda);
                Float4 w1lambda_v = Float4(w1lambda);
                Float4 h0lambda_v = Float4(h0lambda);
                Float4 h1lambda_v = Float4(h1lambda);
                Float4 f00_v      = Float4::load(x_data_ptr);
                Float4 f01_v      = Float4::load(x_data_ptr + w1p * 4);
                Float4 f10_v      = Float4::load(x_data_ptr + h1p * input_width * 4);
                Float4 f11_v      = Float4::load(x_data_ptr + h1p * input_width * 4 + w1p * 4);
                Float4 dst_v      = h0lambda_v * w0lambda_v * f00_v + h0lambda_v * w1lambda_v * f01_v +
                               h1lambda_v * w0lambda_v * f10_v + h1lambda_v * w1lambda_v * f11_v;
                Float4::save(y_data_ptr, dst_v);
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
