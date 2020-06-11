//  Copyright © 2019年 tencent. All rights reserved.

#ifndef TNN_CUDA_UTILS_H
#define TNN_CUDA_UTILS_H

#include "core/common.h"
#include "core/status.h"
#include "core/blob.h"
#include "device/cuda/cuda_common.h"
#include "interpreter/layer_param.h"

namespace TNN_NS {

Status FetchDimensions(Blob * input, Blob * output, CudaLayerBlobInfo &info);

Status FetchKernelInfo(ConvLayerParam * param, CudaLayerKernelInfo &info);

Status FetchKernelInfo(PoolingLayerParam * param, CudaLayerKernelInfo &info);

Status DimsVectorTo4DArray(int * shape, DimsVector dims);

} // tnn

#endif // TNN_CUDA_UTILS_H
