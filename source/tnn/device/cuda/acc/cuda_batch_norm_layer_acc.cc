// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_batch_norm_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

Status CudaBatchNormLayerAcc::Init(Context *context, LayerParam *param,
                                   LayerResource *resource,
                                   const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);

    BatchNormLayerResource *bn_resource =
        dynamic_cast<BatchNormLayerResource *>(resource);

    float *k = bn_resource->k_handle.force_to<float *>();
    float *b = bn_resource->k_handle.force_to<float *>();

    k_size_ = bn_resource->k_handle.GetBytesSize() / sizeof(float);
    b_size_ = bn_resource->b_handle.GetBytesSize() / sizeof(float);

    CUDA_CHECK(cudaMalloc((void **)&k_, k_size_));
    CUDA_CHECK(cudaMalloc((void **)&b_, b_size_));

    CUDA_CHECK(
        cudaMemcpy(k_, k, k_size_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(b_, b, b_size_ * sizeof(float), cudaMemcpyHostToDevice));

    return this->Reshape(inputs, outputs);
}

CudaBatchNormLayerAcc::~CudaBatchNormLayerAcc() {
    if (k_ != nullptr) {
        CUDA_CHECK(cudaFree(k_));
    }
    if (b_ != nullptr) {
        CUDA_CHECK(cudaFree(b_));
    }
}

Status CudaBatchNormLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    FetchDimensions(inputs[0], outputs[0], blob_info_);

    c_div_.init(blob_info_.input_c);
    hw_div_.init(blob_info_.input_h * blob_info_.input_w);

    return TNN_OK;
}

CudaTypeLayerAccRegister<TypeLayerAccCreator<CudaBatchNormLayerAcc>>
    g_cuda_batch_norm_layer_acc_register(LAYER_BATCH_NORM);

}  // namespace TNN_NS
