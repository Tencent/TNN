//  Copyright © 2019年 tencent. All rights reserved.

#ifndef TNN_CUDA_COMMON_H_
#define TNN_CUDA_COMMON_H_

#include "device/cuda/cuda_int_fast_div.h"

namespace TNN_NS {

struct CudaLayerBlobInfo {
    int batch;

    int input_c;
    int input_d;
    int input_h;
    int input_w;

    int output_c;
    int output_d;
    int output_h;
    int output_w;

};  // CudaLayerCommonParams

struct CudaLayerKernelInfo {
    int groups;

    int kernel_d;
    int kernel_h;
    int kernel_w;

    int pad_f;  // D begin
    int pad_e;  // D end
    int pad_t;  // H begin
    int pad_b;  // H end
    int pad_l;  // W begin
    int pad_r;  // W end

    int stride_d;
    int stride_h;
    int stride_w;

    int dilation_d;
    int dilation_h;
    int dilation_w;

};  // CudaLayerROIInfo

struct CudaMemory {
    void* ptr;
    size_t size_in_bytes;
};  // CudaMemory

}  // namespace TNN_NS

#endif  // TNN_CUDA_COMMON_H_
