// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_softmax_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

__global__ void kernel_channel_max(const int num, const int channels, 
    const int spatial_dim, const float* data, float* out) {
    CUDA_KERNEL_LOOP(index, num*spatial_dim) {
        int n = index / spatial_dim;
        int s = index % spatial_dim;
        float maxval = -FLT_MAX;
        for (int c = 0; c < channels; ++c) {
            maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
        }
        out[index] = maxval;
    }
}

__global__ void kernel_channel_subtract_exp(const int count,
    const int num, const int channels,
    const int spatial_dim, const float* bottom_data, const float* channel_max, float* top_data) {
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / channels / spatial_dim;
        int s = index % spatial_dim;
        top_data[index] = exp(bottom_data[index] - channel_max[n * spatial_dim + s]);
    }
}

__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const float* data, float* channel_sum) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
        int n = index / spatial_dim;
        int s = index % spatial_dim;
        float sum = 0;
        for (int c = 0; c < channels; ++c) {
            sum += data[(n * channels + c) * spatial_dim + s];
        }
        channel_sum[index] = sum;
    }
}

__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const float* channel_sum, float* data) {
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / channels / spatial_dim;
        int s = index % spatial_dim;
        data[index] /= channel_sum[n * spatial_dim + s];
    }
}

Status CudaSoftmaxLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
  
    float* bottom_data  = (float*) inputs[0]->GetHandle().base;
    float* top_data     = (float*) outputs[0]->GetHandle().base;

    DimsVector input_dims  = inputs[ 0 ]->GetBlobDesc().dims;

    int channels  = input_dims[axis_];
    int count     = DimsVectorUtils::Count(input_dims); 

    kernel_channel_max<<<RPD_GET_BLOCKS(outer_dim_ * inner_dim_), RPD_CUDA_NUM_THREADS, 0, context_->stream_>>>
      (outer_dim_, channels, inner_dim_, bottom_data, workspace_);

    kernel_channel_subtract_exp<<<RPD_GET_BLOCKS(count), RPD_CUDA_NUM_THREADS, 0, context_->stream_>>>
      (count, outer_dim_, channels, inner_dim_, bottom_data, workspace_, top_data);

    kernel_channel_sum<<<RPD_GET_BLOCKS(outer_dim_ * inner_dim_), RPD_CUDA_NUM_THREADS, 0, context_->stream_>>>
      (outer_dim_, channels, inner_dim_, top_data, workspace_);

    kernel_channel_div<<<RPD_GET_BLOCKS(count), RPD_CUDA_NUM_THREADS, 0, context_->stream_>>>
      (count, outer_dim_, channels, inner_dim_, workspace_, top_data);

    return TNN_OK;
}

}  // namespace TNN_NS
