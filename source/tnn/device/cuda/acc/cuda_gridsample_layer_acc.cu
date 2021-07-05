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

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(GridSample, LAYER_GRIDSAMPLE);

__global__ void gridsample_kernel(const float* input_data, const float* grid_data, float* output_data,
        int output_channel_area, int input_channel_area, int grid_area, int channel, int input_height,
        int input_width) {
    const float* grid_ptr = grid_data + blockIdx.y * grid_area;
    const float* input_ptr = input_data + blockIdx.y * input_channel_area * channel;
    float* output_ptr = output_data + blockIdx.y * output_channel_area * channel;

    CUDA_KERNEL_LOOP(index, output_channel_area) {
        float grid_x = (grid_ptr[2*index] + 1) * input_width * 0.5 -0.5;
        float grid_y = (grid_ptr[2*index+1] + 1) * input_height * 0.5 - 0.5;

        const int w0 = floorf(grid_x);
        const int w1p = (w0 < input_width - 1) ? 1 : 0;
        float w1lambda = grid_x - w0;
        float w0lambda = (float)1. - w1lambda;
        if (w0 < 0 || w0 > input_width - 1) {
            w0lambda = 0;
        }
        if (w0 + 1 < 0 || w0 + 1 > input_width - 1) {
            w1lambda = 0;
        }

        const int h0  = floorf(grid_y);
        const int h1p = (h0 < input_height - 1) ? 1 : 0;
        float h1lambda = grid_y - h0;
        float h0lambda = (float)1. - h1lambda;
        if (h0 < 0 || h0 > input_height - 1) {
            h0lambda = 0;
        }
        if (h0 + 1 < 0 || h0 + 1 > input_height - 1) {
            h1lambda = 0;
        }

        const float *x_data_ptr = input_ptr + h0 * input_width + w0;
        float *y_data_ptr = output_ptr + index;
        for (int c=0; c<channel; c++) {
            y_data_ptr[0] = h0lambda * (w0lambda * x_data_ptr[0] + w1lambda * x_data_ptr[w1p]) +
                            h1lambda * (w0lambda * x_data_ptr[h1p * input_width] +
                                        w1lambda * x_data_ptr[h1p * input_width + w1p]);

            x_data_ptr += input_channel_area;
            y_data_ptr += output_channel_area;
        }
    }
}

Status CudaGridSampleLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaGridSampleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaGridSampleLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_dims = inputs[0]->GetBlobDesc().dims;
    auto grid_dims = inputs[1]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    int channel = input_dims[1];
    int input_height = input_dims[2];
    int input_width = input_dims[3];
    int input_channel_area = DimsVectorUtils::Count(input_dims, 2);
    int output_channel_area = DimsVectorUtils::Count(output_dims, 2);
    int grid_area = DimsVectorUtils::Count(grid_dims, 1);

    float* input_data  = (float *)((char *)inputs[0]->GetHandle().base+ inputs[0]->GetHandle().bytes_offset);
    float* grid_data   = (float *)((char *)inputs[1]->GetHandle().base+ inputs[1]->GetHandle().bytes_offset);
    float* output_data = (float *)((char *)outputs[0]->GetHandle().base + outputs[0]->GetHandle().bytes_offset);

    dim3 griddim(TNN_CUDA_GET_BLOCKS(output_channel_area), input_dims[0]);
    gridsample_kernel<<<griddim, TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(input_data, grid_data, output_data,
        output_channel_area, input_channel_area, grid_area, channel, input_height, input_width);

    return TNN_OK;
}

REGISTER_CUDA_ACC(GridSample, LAYER_GRIDSAMPLE);

}  // namespace TNN_NS
