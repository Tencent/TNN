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

Status CpuGridSampleLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<GridSampleLayerParam *>(param_);
    
    auto input_dims = inputs[0]->GetBlobDesc().dims;
    auto grid_dims = inputs[1]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    
    if (output_dims.size() != 4) {
        return Status(TNNERR_PARAM_ERR, "CpuGridSampleLayerAcc only support 4D sampler");
    }
    
    if (layer_param->mode != 2 || layer_param->pad_type != 0 || layer_param->align_corners != 0) {
        return Status(TNNERR_PARAM_ERR, "CpuGridSampleLayerAcc dont support some mode or pade type or align_corners");
    }
    
    const int batch = input_dims[0];
    const int channel = input_dims[1];
    const int input_channel_area = DimsVectorUtils::Count(input_dims, 2);
    const int output_channel_area = DimsVectorUtils::Count(output_dims, 2);
    float *input_data_ptr  = (float *)((char *)inputs[0]->GetHandle().base+ inputs[0]->GetHandle().bytes_offset);
    float *grid_data_ptr  = (float *)((char *)inputs[1]->GetHandle().base+ inputs[1]->GetHandle().bytes_offset);
    float *output_data_ptr = (float *)((char *)outputs[0]->GetHandle().base + outputs[0]->GetHandle().bytes_offset);
    
    const int grid_area = DimsVectorUtils::Count(grid_dims, 1);
    
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        const int input_height = input_dims[2];
        const int input_width = input_dims[3];
        for (int b=0; b<batch; b++) {
            auto grid_data_b = grid_data_ptr + b*grid_area;
            auto input_data_b = input_data_ptr + b*input_channel_area*channel;
            auto output_data_b = output_data_ptr + b*output_channel_area*channel;
            
            for (int i=0; i<output_channel_area; i++) {
                auto grid_data = grid_data_b + 2*i;
                float grid_x = (grid_data[0] + 1) * input_width * 0.5 -0.5;
                float grid_y = (grid_data[1] + 1) * input_height * 0.5 - 0.5;
                
                const int w0 = std::floor(grid_x);
                const int w1p = (w0 < input_width - 1) ? 1 : 0;
                float w1lambda = grid_x - w0;
                float w0lambda = (float)1. - w1lambda;
                if (w0 < 0 || w0 > input_width - 1) {
                    w0lambda = 0;
                }
                if (w0 + 1 < 0 || w0 + 1 > input_width - 1) {
                    w1lambda = 0;
                }
                
                const int h0  = std::floor(grid_y);
                const int h1p = (h0 < input_height - 1) ? 1 : 0;
                float h1lambda = grid_y - h0;
                float h0lambda = (float)1. - h1lambda;
                if (h0 < 0 || h0 > input_height - 1) {
                    h0lambda = 0;
                }
                if (h0 + 1 < 0 || h0 + 1 > input_height - 1) {
                    h1lambda = 0;
                }
                LOGD("h0: %d, input_width:%d, w0:%d", h0, input_width, w0);
                //Note: read outside valid roi will raise
                const float *x_data_ptr = input_data_b + h0 * input_width + w0;
                if (x_data_ptr < input_data_ptr) {
                    x_data_ptr = input_data_ptr;
                }
                float *y_data_ptr = output_data_b + i;
                for (int c=0; c<channel; c++) {
                    y_data_ptr[0] = h0lambda * (w0lambda * x_data_ptr[0] + w1lambda * x_data_ptr[w1p]) +
                                    h1lambda * (w0lambda * x_data_ptr[h1p * input_width] +
                                                w1lambda * x_data_ptr[h1p * input_width + w1p]);
                    
                    x_data_ptr += input_channel_area;
                    y_data_ptr += output_channel_area;
                }
            }
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "CpuGridSampleLayerAcc now only support float data");
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(GridSample, LAYER_GRIDSAMPLE);

}  // namespace TNN_NS
