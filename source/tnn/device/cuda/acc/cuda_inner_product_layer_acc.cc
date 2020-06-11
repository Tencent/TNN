// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_inner_product_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

Status CudaInnerProductLayerAcc::Init(Context *context, LayerParam *param,
                                      LayerResource *resource,
                                      const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);

    InnerProductLayerParam *ip_param =
        dynamic_cast<InnerProductLayerParam *>(param);
    if (ip_param == nullptr) {
        LOGE("Convert to InnerProductLayerParam failed\n");
        return TNNERR_LAYER_ERR;
    }

    InnerProductLayerResource *ip_resource =
        dynamic_cast<InnerProductLayerResource *>(resource);
    if (ip_resource == nullptr) {
        LOGE("Convert to InnerProductLayerResource failed\n");
        return TNNERR_LAYER_ERR;
    }

    has_bias_        = ip_param->has_bias;
    multiplier_size_ = 0;

    float *weight = ip_resource->weight_handle.force_to<float *>();
    float *bias   = ip_resource->bias_handle.force_to<float *>();

    weight_size_ = ip_resource->weight_handle.GetBytesSize() / sizeof(float);
    bias_size_   = ip_resource->bias_handle.GetBytesSize() / sizeof(float);

    CUDA_CHECK(cudaMalloc((void **)&weight_, weight_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&bias_, bias_size_ * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(weight_, weight, weight_size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bias_, bias, bias_size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    return this->Reshape(inputs, outputs);
}

CudaInnerProductLayerAcc::~CudaInnerProductLayerAcc() {
    if (weight_ != nullptr) {
        CUDA_CHECK(cudaFree(weight_));
        weight_ = nullptr;
    }
    if (bias_ != nullptr) {
        CUDA_CHECK(cudaFree(bias_));
        bias_ = nullptr;
    }
    if (multiplier_ != nullptr) {
        CUDA_CHECK(cudaFree(multiplier_));
        multiplier_ = nullptr;
    }
}

Status CudaInnerProductLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *ip_param =
        dynamic_cast<InnerProductLayerParam *>(param_);

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];

    DimsVector input_dims  = input_blob->GetBlobDesc().dims;
    DimsVector output_dims = output_blob->GetBlobDesc().dims;

    FetchDimensions(input_blob, output_blob, blob_info_);

    n_       = ip_param->num_output;
    int axis = ip_param->axis;
    m_       = DimsVectorUtils::Count(input_dims, 0, axis);
    k_       = DimsVectorUtils::Count(input_dims, axis);

    if (k_ * n_ != weight_size_) {
        LOGE("weight size (%lu) != N(%d) * K(%d). \n", weight_size_, n_, k_);
        return TNNERR_LAYER_ERR;
    }

    if (has_bias_) {
        if (m_ > multiplier_size_) {
            multiplier_size_ = m_;
            if (multiplier_ != nullptr) {
                CUDA_CHECK(cudaFree(multiplier_));
                multiplier_ = nullptr;
            }
            CUDA_CHECK(cudaMalloc((void **)&multiplier_,
                                  multiplier_size_ * sizeof(float)));
            float *tmp = new float[multiplier_size_];
            for (int i = 0; i < multiplier_size_; i++) {
                tmp[i] = 1.0;
            }
            CUDA_CHECK(cudaMemcpy(multiplier_, tmp,
                                  multiplier_size_ * sizeof(float),
                                  cudaMemcpyHostToDevice));
            delete[] tmp;
        }
    }

    return TNN_OK;
}

Status CudaInnerProductLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *ip_param =
        dynamic_cast<InnerProductLayerParam *>(param_);

    float *bottom_data = (float *)inputs[0]->GetHandle().base;
    float *top_data    = (float *)outputs[0]->GetHandle().base;

    float alpha = 1.0;
    float beta  = 0.0;

    CUBLAS_CHECK(cublasSgemm(context_->cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                             n_, m_, k_, &alpha, weight_, k_, bottom_data, k_,
                             &beta, top_data, n_));

    if (has_bias_) {
        alpha = 1.0;
        beta  = 1.0;
        CUBLAS_CHECK(cublasSgemm(context_->cublas_handle_, CUBLAS_OP_N,
                                 CUBLAS_OP_N, n_, m_, 1, &alpha, bias_, n_,
                                 multiplier_, 1, &beta, top_data, n_));
    }

    return TNN_OK;
}

CudaTypeLayerAccRegister<TypeLayerAccCreator<CudaInnerProductLayerAcc>>
    g_cuda_inner_product_layer_acc_register(LAYER_INNER_PRODUCT);

}  // namespace TNN_NS
