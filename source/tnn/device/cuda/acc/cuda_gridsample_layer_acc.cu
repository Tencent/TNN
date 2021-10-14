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

static __forceinline__ __device__
bool within_bounds_2d(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

__global__ void gridsample_kernel(const float* input_data, const float* grid_data, float* output_data,
        int output_channel_area, int input_channel_area, int grid_area, int channel, int input_height,
        int input_width) {
    const float* grid_ptr = grid_data + blockIdx.y * grid_area;
    const float* input_ptr = input_data + blockIdx.y * input_channel_area * channel;
    float* output_ptr = output_data + blockIdx.y * output_channel_area * channel;

    CUDA_KERNEL_LOOP(index, output_channel_area) {
        float ix = (grid_ptr[2*index] + 1) * input_width * 0.5 -0.5;
        float iy = (grid_ptr[2*index+1] + 1) * input_height * 0.5 - 0.5;
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
        const float *input_c = input_ptr;
        float *output_c = output_ptr + index;
        for (int c = 0; c < channel;
                ++c, output_c += output_channel_area, input_c += input_channel_area) {
            auto res = static_cast<float>(0);
            res += input_c[nw_index] * nw;
            res += input_c[ne_index] * ne;
            res += input_c[sw_index] * sw;
            res += input_c[se_index] * se;
            *output_c = res;
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
