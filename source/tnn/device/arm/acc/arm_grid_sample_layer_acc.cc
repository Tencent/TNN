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

template <typename T>
static void Compute(T *input_data, T *grid_data, T *output_data, DimsVector &input_dims, DimsVector &output_dims) {
    auto channel             = input_dims[1];
    auto input_height        = input_dims[2];
    auto input_width         = input_dims[3];
    auto input_channel_area  = DimsVectorUtils::Count(input_dims, 2);
    auto output_channel_area = DimsVectorUtils::Count(output_dims, 2);
    OMP_PARALLEL_FOR_
    for (int i = 0; i < output_channel_area; ++i) {
        auto grid_position = grid_data + 2 * i;
        T grid_x           = (grid_position[0] + 1) * input_width * 0.5 - 0.5;
        T grid_y           = (grid_position[1] + 1) * input_height * 0.5 - 0.5;
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
            // f(x,y) = (1-y) * (1-x) * f(0, 0) +  (1-y) * (x) * f(1, 0) + (y) * (1-x) * f(0, 1) + (x) * (y) * f(1, 1)
            y_data_ptr[0] = h0lambda * w0lambda * x_data_ptr[0] + h0lambda * w1lambda * x_data_ptr[w1p] +
                            h1lambda * w0lambda * x_data_ptr[h1p * input_width] +
                            h1lambda * w1lambda * x_data_ptr[h1p * input_width + w1p];
            x_data_ptr += input_channel_area;
            y_data_ptr += output_channel_area;
        }
    }
}

Status ArmGridSampleLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param       = dynamic_cast<GridSampleLayerParam *>(param_);
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto grid_dims   = inputs[1]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    if (!(input_dims.size() == 4 && param->mode == 2 && param->pad_type == 0 && param->align_corners == 0)) {
        LOGE("Error: Arm layer acc don't support GridSample input size(%lu) or param:(%d, %d, %d)\n", input_dims.size(),
             param->mode, param->pad_type, param->align_corners);
        return Status(TNNERR_MODEL_ERR, "Error: Arm layer acc don't support.\n");
    }
    auto batch               = input_dims[0];
    auto channel             = input_dims[1];
    auto input_channel_area  = DimsVectorUtils::Count(input_dims, 2);
    auto grid_area           = DimsVectorUtils::Count(grid_dims, 1);
    auto output_channel_area = DimsVectorUtils::Count(output_dims, 2);
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_base_ptr  = (float *)((char *)inputs[0]->GetHandle().base + inputs[0]->GetHandle().bytes_offset);
        float *grid_base_ptr   = (float *)((char *)inputs[1]->GetHandle().base + inputs[1]->GetHandle().bytes_offset);
        float *output_base_ptr = (float *)((char *)outputs[0]->GetHandle().base + outputs[0]->GetHandle().bytes_offset);
        for (int n = 0; n < batch; n++) {
            auto input_data  = input_base_ptr + n * channel * input_channel_area;
            auto grid_data   = grid_base_ptr + n * grid_area;
            auto output_data = output_base_ptr + n * channel * output_channel_area;
            Compute<float>(input_data, grid_data, output_data, input_dims, output_dims);
        }
    } else {
        LOGE("Error: Arm layer acc don't support datatype: %d\n", inputs[0]->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: Arm layer acc don't support datatype\n");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(GridSample, LAYER_GRIDSAMPLE);
REGISTER_ARM_LAYOUT(LAYER_GRIDSAMPLE, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
