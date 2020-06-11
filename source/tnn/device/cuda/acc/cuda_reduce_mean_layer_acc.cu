// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_reduce_mean_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {
        

__global__ void kernel_reduce_mean(const int num, const int channels,
    const int spatial_dim, const float* input, float* output) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
        int n = index / spatial_dim;
        int s = index % spatial_dim;
        float sum = 0;
        for (int c = 0; c < channels; ++c) {
            sum += input[(n * channels + c) * spatial_dim + s];
        }
        output[n * spatial_dim + s] = sum / float(channels);
    }
}

Status CudaReduceMeanLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                                                 const std::vector<Blob *> &outputs) {

    int channels = inputs[0]->GetBlobDesc().dims[axis_];
    float* bottom_data  = (float*) inputs[0]->GetHandle().base;
    float* top_data     = (float*) outputs[0]->GetHandle().base;
    size_t count = DimsVectorUtils::Count(inputs[ 0 ]->GetBlobDesc().dims); 

    if ( channels == 1) {
      CUDA_CHECK(cudaMemcpyAsync(top_data, bottom_data, count * sizeof(float), cudaMemcpyDeviceToDevice, context_->stream_));
      return TNN_OK;
    } else {
      kernel_reduce_mean<<<RPD_GET_BLOCKS(outer_dim_ * inner_dim_), RPD_CUDA_NUM_THREADS, 0, context_->stream_>>>
        (outer_dim_, channels, inner_dim_, bottom_data, top_data);
    }

    return TNN_OK;
}

}  // namespace TNN_NS
