// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_permute_layer_acc.h"

#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

Status CudaPermuteLayerAcc::Init(Context *context, LayerParam *param,
                                 LayerResource *resource,
                                 const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);

    DimsVector input_dims = inputs[0]->GetBlobDesc().dims;
    n_dims_               = input_dims.size();

    old_steps_.resize(n_dims_);
    new_steps_.resize(n_dims_);

    CUDA_CHECK(cudaMalloc((void **)&permute_order_d_, n_dims_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&old_steps_d_, n_dims_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&new_steps_d_, n_dims_ * sizeof(int)));

    return this->Reshape(inputs, outputs);
}

CudaPermuteLayerAcc::~CudaPermuteLayerAcc() {
    CUDA_CHECK(cudaFree(permute_order_d_));
    CUDA_CHECK(cudaFree(old_steps_d_));
    CUDA_CHECK(cudaFree(new_steps_d_));
}

Status CudaPermuteLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    FetchDimensions(inputs[0], outputs[0], blob_info_);

    PermuteLayerParam *permute_param =
        dynamic_cast<PermuteLayerParam *>(param_);

    for (int i = 0; i < n_dims_; ++i) {
        if (i == n_dims_ - 1) {
            old_steps_[i] = 1;
        } else {
            old_steps_[i] = DimsVectorUtils::Count(input_dims, i + 1);
        }
    }

    for (int i = 0; i < n_dims_; ++i) {
        if (i == n_dims_ - 1) {
            new_steps_[i] = 1;
        } else {
            new_steps_[i] = DimsVectorUtils::Count(output_dims, i + 1);
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(new_steps_d_, new_steps_.data(),
                               n_dims_ * sizeof(int), cudaMemcpyHostToDevice,
                               context_->stream_));

    CUDA_CHECK(cudaMemcpyAsync(old_steps_d_, old_steps_.data(),
                               n_dims_ * sizeof(int), cudaMemcpyHostToDevice,
                               context_->stream_));

    CUDA_CHECK(cudaMemcpyAsync(permute_order_d_, permute_param->orders.data(),
                               n_dims_ * sizeof(int), cudaMemcpyHostToDevice,
                               context_->stream_));

    return TNN_OK;
}

CudaTypeLayerAccRegister<TypeLayerAccCreator<CudaPermuteLayerAcc>>
    g_cuda_permute_layer_acc_register(LAYER_PERMUTE);

}  // namespace TNN_NS
