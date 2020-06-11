// Copyright 2019 Tencent. All Rights Reserved

#include "device/cuda/acc/cuda_relu_layer_acc.h"
#include <iostream>
#include "device/cuda/cuda_utils.h"
#include "utils/dims_vector_utils.h"

namespace TNN_NS {

Status CudaReluLayerAcc::Init(Context *context, LayerParam *param,
                              LayerResource *resource,
                              const std::vector<Blob *> &inputs,
                              const std::vector<Blob *> &outputs) {
    CudaLayerAcc::Init(context, param, resource, inputs, outputs);

    activation_mode_ = CUDNN_ACTIVATION_RELU;
    tensor_format_   = CUDNN_TENSOR_NCHW;
    data_type_       = CUDNN_DATA_FLOAT;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc_));
    CUDNN_CHECK(cudnnSetActivationDescriptor(activation_desc_, activation_mode_,
                                             CUDNN_PROPAGATE_NAN, 1.0));

    alpha_ = 1.0f;
    beta_  = 0.0f;

    return this->Reshape(inputs, outputs);
}

CudaReluLayerAcc::~CudaReluLayerAcc() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activation_desc_));
}

Status CudaReluLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    // LOGD("input n,c,d,h,w: %d, %d, %d, %d, %d , output n,c,h,w: %d, %d,
    // %d, %d, %d \n",
    //      input_dims[0], input_dims[1], input_dims[2], input_dims[3],
    //      input_dims[4], output_dims[0], output_dims[1], output_dims[2],
    //      output_dims[3], output_dims[4]);

    FetchDimensions(inputs[0], outputs[0], blob_info_);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        bottom_desc_, tensor_format_, data_type_, blob_info_.batch,
        blob_info_.input_c, blob_info_.input_d * blob_info_.input_h,
        blob_info_.input_w));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        top_desc_, tensor_format_, data_type_, blob_info_.batch,
        blob_info_.output_c, blob_info_.output_d * blob_info_.output_h,
        blob_info_.output_w));

    return TNN_OK;
}

Status CudaReluLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    CUDNN_CHECK(cudnnActivationForward(
        context_->cudnn_handle_, activation_desc_, &alpha_, bottom_desc_,
        inputs[0]->GetHandle().base, &beta_, top_desc_,
        outputs[0]->GetHandle().base));

    CUDA_CHECK(cudaStreamSynchronize(context_->stream_));

    return TNN_OK;
}

CudaTypeLayerAccRegister<TypeLayerAccCreator<CudaReluLayerAcc>>
    g_cuda_relu_layer_acc_register(LAYER_RELU);

}  // namespace TNN_NS
