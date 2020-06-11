// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_permute_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {
        
__global__ void permute_kernel(int n, const float * srcData,
                               int num_axes, int * permute_order,
                               int * old_steps, int * new_steps,
                               float * dstData) {
    CUDA_KERNEL_LOOP(index, n) {
        int old_idx = 0;
        int idx = index;
        for (int j = 0; j < num_axes; ++j) {
            int order = permute_order[j];
            old_idx += (idx / new_steps[j]) * old_steps[order];
            idx %= new_steps[j];
        }
        dstData[index] = srcData[old_idx];
    }
}

Status CudaPermuteLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                                                 const std::vector<Blob *> &outputs) {

    float* bottom_data  = (float*) inputs[0]->GetHandle().base;
    float* top_data     = (float*) outputs[0]->GetHandle().base;
    size_t count = DimsVectorUtils::Count(inputs[ 0 ]->GetBlobDesc().dims); 

    permute_kernel<<<RPD_GET_BLOCKS(count), RPD_CUDA_NUM_THREADS, 0, context_->stream_>>>
          (count, bottom_data, n_dims_, 
          permute_order_d_, old_steps_d_, new_steps_d_, top_data);

    return TNN_OK;
}

}  // namespace TNN_NS
