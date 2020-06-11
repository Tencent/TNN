// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_batch_norm_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

template<int THREAD_PER_BLOCK, int ELE_PER_THREAD>
__global__ void bn_relu_unroll_kernel ( 
    int_fastdiv channels, const int chw,
    int_fastdiv hw, const float* src_data, 
    const float* k, const float* b,float *dst, 
    bool relu, const float relu_negative_slope) 
{
    const int n_off = blockIdx.y * chw;
    const int ele_off = ELE_PER_THREAD * THREAD_PER_BLOCK * blockIdx.x + threadIdx.x;

    src_data += n_off + ele_off;
    dst      += n_off + ele_off;
    int index = n_off + ele_off;

    #pragma unroll
    for(int i=0; i<ELE_PER_THREAD; ++i){
            if(ele_off + i*THREAD_PER_BLOCK < chw){
                const int c = (index / hw) % channels;
                float val = src_data[i*THREAD_PER_BLOCK] * k[c] + b[c];
                if(relu) val = val>=0?val:val*relu_negative_slope;
                dst[i*THREAD_PER_BLOCK] = val;
            }
            index += THREAD_PER_BLOCK;
    }

}


void bn_relu_launcher(int n, int c, int hw, const float *src, 
                                            const float *k, const float *b, float *dst, bool relu, 
                                            const int_fastdiv &cdiv, const int_fastdiv &hwdiv,
                                            const float relu_negative_slope, cudaStream_t *stream){
    const int THREAD_PER_BLOCK = 128;
    const int ELE_PER_THREAD = 4; 
    dim3 griddim;
    griddim.x = (c*hw + ELE_PER_THREAD * THREAD_PER_BLOCK - 1) /(ELE_PER_THREAD * THREAD_PER_BLOCK);
    griddim.y = n;
    bn_relu_unroll_kernel<THREAD_PER_BLOCK,ELE_PER_THREAD><<<griddim, 128,0, *stream>>>
                    (cdiv, c*hw, hwdiv, src, k, b,dst, relu, relu_negative_slope);
}


Status CudaBatchNormLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                          const std::vector<Blob *> &outputs) {

    const int hw = blob_info_.input_h * blob_info_.input_w;
    bn_relu_launcher(blob_info_.batch, blob_info_.input_c, hw,
                     (float*) inputs[0]->GetHandle().base,
                     k_, b_, 
                     (float*) outputs[0]->GetHandle().base,
                     false, 
                     c_div_, hw_div_, 0.f, &context_->stream_);
    return TNN_OK;
}

}
