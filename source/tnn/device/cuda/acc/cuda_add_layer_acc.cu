// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_add_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

template<int THREAD_PER_BLOCK, int ELE_PER_THREAD>
__global__ void add_kernel (
            const int channels,
            const int dim,
            const float * src0,
            const float * src1,
            const float* weights,   
            float *dst_data) 
{
    
    int chw = channels * dim;
    int n_off = blockIdx.y * chw;
    int ele_off = ELE_PER_THREAD * THREAD_PER_BLOCK * blockIdx.x + threadIdx.x;

    dst_data += n_off + ele_off;

    float val[ELE_PER_THREAD] = {0};
    
    const float * __restrict__ a_ptr = src0 + n_off + ele_off;
    const float * __restrict__ b_ptr = src1 + n_off + ele_off;
    #pragma unroll
    for(int i=0; i<ELE_PER_THREAD; ++i){
            if(ele_off + i*THREAD_PER_BLOCK < chw)
                val[i] = a_ptr[i*THREAD_PER_BLOCK] + 
                         b_ptr[i*THREAD_PER_BLOCK];
    }

    #pragma unroll
    for(int i=0; i<ELE_PER_THREAD; ++i){
            if(ele_off + i*THREAD_PER_BLOCK < chw)
            {
                    if(weights == NULL)
                    {
                            dst_data[i*THREAD_PER_BLOCK] = val[i];
                    } 
                    else
                    {
                            int channel = ((ele_off + i*THREAD_PER_BLOCK) / dim) % channels;
                            dst_data[i*THREAD_PER_BLOCK] = val[i] + weights[channel];
                    }
            }
    }
}

Status CudaAddLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {

    int channels = blob_info_.input_c;
    int dim = blob_info_.input_d * blob_info_.input_h * blob_info_.input_w;

    const int THREAD_PER_BLOCK = 128;
    const int ELE_PER_THREAD = 16;
    dim3 griddim;
    griddim.x = (channels * dim + ELE_PER_THREAD * THREAD_PER_BLOCK - 1) /(ELE_PER_THREAD * THREAD_PER_BLOCK);
    griddim.y = blob_info_.batch;

    add_kernel<THREAD_PER_BLOCK,ELE_PER_THREAD><<<griddim, THREAD_PER_BLOCK, 0, context_->stream_>>>
                    (channels, dim, 
                     (float*)inputs[0]->GetHandle().base,
                     (float*)inputs[1]->GetHandle().base,
                     nullptr, 
                     (float*)outputs[0]->GetHandle().base);

    return TNN_OK;
}


}  // namespace TNN_NS
